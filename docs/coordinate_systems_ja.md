# voxcity の座標系・測地系の扱い

VoxCity パッケージにおける座標系（CRS）と測地系の扱いを調査してまとめたもの。

## 1. 基本方針：内部は WGS84（EPSG:4326）経緯度に統一

パッケージ全体を貫く設計は「**すべてのデータを WGS84 の経度・緯度（lon, lat）で保持する**」というもの。ほとんどのデータソースのダウンローダーは、出力を EPSG:4326 に揃える。

- OSM: [osm.py:796](../src/voxcity/downloader/osm.py#L796) など `crs="EPSG:4326"`
- EUBUCCO: [eubucco.py:336-340](../src/voxcity/downloader/eubucco.py#L336-L340) 入力CRSが異なれば `to_crs(4326)`
- GBA: [gba.py:128-131](../src/voxcity/downloader/gba.py#L128-L131)
- GEE: [gee.py:644](../src/voxcity/downloader/gee.py#L644) `crs='EPSG:4326'` でエクスポート

領域指定の `rectangle_vertices` も `(lon, lat)` の WGS84 前提。

**例外（PLATEAU/CityGML）**：[citygml.py:960-974](../src/voxcity/downloader/citygml.py#L960-L974) では `set_crs(epsg=6697)` のまま座標順序の入れ替え（`swap_coordinates_if_needed`）だけ行い、**`to_crs(4326)` による再投影はしない**。JGD2011 の経緯度値をそのまま lon/lat として下流に渡す（→ 第7節のデータム非変換の具体例）。両者は数値的にサブメートルで近接するため、街区スケールでは実害が出にくい前提。

## 2. 測地計算は WGS84 楕円体上の測地線（pyproj.Geod）

距離・スケール計算は球面近似ではなく **WGS84 楕円体上の測地線**で行う。

- [utils/__init__.py:64](../src/voxcity/utils/__init__.py#L64) でモジュール共有のシングルトン `_WGS84_GEOD = Geod(ellps='WGS84')` を定義（ステートレス・スレッドセーフ）
- [calculate_distance](../src/voxcity/utils/__init__.py#L185) は `geod.inv()`（逆測地計算）で楕円体測地線距離を返す（great-circle より正確、とdocstringに明記）
- [normalize_to_one_meter](../src/voxcity/utils/__init__.py#L208) で「1メートルあたりの経緯度（度）ベクトル」を作り、メートル↔度の局所換算に使う
- 別途、球面近似の `haversine_distance` も補助的に存在

## 3. ローカルグリッド座標系（uv_m）とアフィン投影

シミュレーション格子は、緯度経度を直接使わず**グリッド原点からのメートル系（uv_m）**を持つ。中心は [utils/projector.py](../src/voxcity/utils/projector.py) の `GridProjector`。

- 2つの座標系を定義：
  - `lon_lat`：地理座標（WGS84）
  - `uv_m`：グリッド原点（`rectangle_vertices[0]`）からのメートル（u=side_1方向、v=side_2方向）
- 変換は **2×2 アフィン行列**（[projector.py:59-89](../src/voxcity/utils/projector.py#L59-L89)）。逆行列を事前計算して O(1) 変換
- 不変条件：`voxcity_grid[i,j,k] ↔ シーン座標 (i·meshsize, j·meshsize, k·meshsize)`

グリッドジオメトリの計算は [raster/core.py の compute_grid_geometry](../src/voxcity/geoprocessor/raster/core.py#L153)：
- 四辺形の辺ベクトル（lon/lat度）を、楕円体測地線距離（`calculate_distance`）で割って `u_vec`/`v_vec`（度/メートル）に正規化
- [calculate_grid_size](../src/voxcity/geoprocessor/utils.py#L92) でメッシュ数と調整済みメッシュサイズを算出

→ **メッシュ寸法は楕円体測地線で正確に決まる**が、格子自体は lon/lat 空間の平行四辺形（アフィン）なので、局所的には正確・広域では歪みが残る設計。

## 4. CRS 変換ユーティリティ

[geoprocessor/utils.py](../src/voxcity/geoprocessor/utils.py) に汎用変換が集約：

- [setup_transformer](../src/voxcity/geoprocessor/utils.py#L228) / `transform_coords`：`pyproj.Transformer.from_crs(..., always_xy=True)`。**結果をキャッシュ**して再構築コストを回避
- [normalize_gdf_crs](../src/voxcity/geoprocessor/raster/core.py#L239)：GeoDataFrame を 4326 に揃える。**CRS が無い場合は EPSG:4326 と仮定して警告**

`always_xy=True` を一貫して使い、軸順（lon-lat vs lat-lon）の混乱を抑えている。

## 5. ラスタ／DEM 処理での動的投影

GeoTIFF を読むときはファイル自身の CRS を尊重し、必要時のみ変換する。

- 高さグリッド ([raster/raster.py:33-37](../src/voxcity/geoprocessor/raster/raster.py#L33-L37))：`src.crs` が 4326 でなければセル中心をラスタCRSへ Transformer 変換してサンプリング
- **DEM 補間** ([create_dem_grid_from_geotiff_polygon](../src/voxcity/geoprocessor/raster/raster.py#L51))：補間を歪みなく行うため、領域中心経緯度から **UTM ゾーンを動的算出**（北半球 `32600+zone`／南半球 `32700+zone`）し、メートル系で `griddata` 補間
- 樹冠（canopy）処理も同様に UTM へ投影して距離計算（[canopy.py:145-147](../src/voxcity/geoprocessor/raster/canopy.py#L145-L147)）

## 6. データソース別のネイティブ CRS

| ソース | ネイティブCRS | 扱い |
|---|---|---|
| GSI DEM（国土地理院） | **EPSG:3857**（Web Mercator）で書き出し | 下流の DEM グリッド生成で再投影（[gsi.py:12-13](../src/voxcity/downloader/gsi.py#L12-L13)） |
| OEMJ | EPSG:3857 | [oemj.py:290-292](../src/voxcity/downloader/oemj.py#L290-L292) |
| PLATEAU/CityGML | **EPSG:6697**（JGD2011 経緯度＋標高） | 再投影せず 6697 のまま。lat-lon 順の入れ替えのみ（[citygml.py:829-854](../src/voxcity/downloader/citygml.py#L829-L854), [960-973](../src/voxcity/downloader/citygml.py#L960-L973)） |
| GEE | EPSG:4326 | エクスポート時に指定 |
| OSM バッファ処理 | Albers正積（AEA）へ一時投影 | [osm.py:1056-1062](../src/voxcity/downloader/osm.py#L1056-L1062) |

## 7. 日本の測地系まわり

- **EPSG:6697 = JGD2011 地理座標系 ＋ 標高**の複合CRS（PLATEAU 用）として扱う。座標が緯度経度逆順で入ることを想定した `swap_coordinates_if_needed` を用意
- **GSI DEM** は基盤地図情報の標高を Web Mercator GeoTIFF として保存し、後段で再投影
- ただしコード上は **JGD2011 ⇔ WGS84 のデータム変換を明示的には実装しておらず、pyproj に委ねている**。実用上 JGD2011（ITRF系）と WGS84 はサブメートルで近接するため、街区スケールでは問題になりにくい前提

## 8. 鉛直方向（高さ・標高）の扱い

- DEM は標高値としてそのまま使い、**最小値を 0 にオフセット**して地面基準にする（[process_grid](../src/voxcity/geoprocessor/utils.py#L75)、[importer/transform.py:128-129](../src/voxcity/importer/transform.py#L128-L129) の `dem_min` を減算）
- **ジオイド高 ⇔ 楕円体高の変換は行っていない**。標高（正標高）をそのままボクセルの地盤高として採用
- ボクセル化では地盤の1つ上にモデルを載せる `+1` オフセット（`ground_level = int(dem/voxel_size+0.5)+1`）

## 9. 格子の向き（orientation）

座標系とは別に格子の行方向の規約がある（[utils/orientation.py](../src/voxcity/utils/orientation.py)）：

- 内部処理は **south_up**（行0＝南＝原点辺、行increase で北へ／列increase で東へ）
- 可視化時のみ `np.flipud` で north_up に反転
- 変換は `ensure_orientation()` のみで行う（境界での正規化用）

---

## まとめ（要点）

1. **保持形式は常に WGS84 経緯度**。投影座標系では保持しない
2. **測地計算は WGS84 楕円体測地線**（`pyproj.Geod`）で正確
3. **シミュレーション格子は原点基準のローカルメートル系（uv_m）**、lon/lat とは 2×2 アフィンで相互変換
4. **DEM／ラスタ補間時のみ UTM へ動的投影**して歪みを抑制
5. データソースのネイティブCRS（3857, 6697 等）は読み込み境界で吸収し、内部 4326 に正規化
6. **データム変換（特に日本の JGD ⇔ WGS84）と鉛直系（ジオイド）は明示処理せず pyproj／標高そのままに依存** ― これが精度上の主な注意点
