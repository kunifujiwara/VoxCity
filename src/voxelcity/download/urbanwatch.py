import os
import tempfile

from .utils import download_file_google_drive
from ..geo.utils import haversine_distance, raster_intersects_polygon, save_raster, create_polygon, create_geodataframe, merge_geotiffs

def process_city(city, city_data, polygon, distance_threshold, output_dir):
    city_lat, city_lon = city_data["coords"]
    centroid = polygon.centroid
    distance = haversine_distance(centroid.y, centroid.x, city_lat, city_lon)

    if distance <= distance_threshold:
        print(f"Processing {city} (distance: {distance:.2f} km)")
        return process_city_files(city_data, polygon, output_dir)
    else:
        # print(f"Skipping {city} (distance: {distance:.2f} km)")
        return []

def process_city_files(city_data, polygon, output_dir):
    geotiff_files = []
    for filename, file_id in city_data["files"].items():
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            temp_path = temp_file.name

        if download_file_google_drive(file_id, temp_path):
            try:
                # print(f"Checking intersection for {filename}")
                if raster_intersects_polygon(temp_path, polygon):
                    geotiff_path = os.path.join(output_dir, f"{filename}")
                    save_raster(temp_path, geotiff_path)
                    geotiff_files.append(geotiff_path)
                    print(f"File processed: {filename}")
                # else:
                #     print(f"File does not intersect with the polygon: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            finally:
                os.unlink(temp_path)
        else:
            print(f"Failed to download: {filename}")
    return geotiff_files

def get_geotif_urbanwatch(rotated_rectangle_vertices, output_dir):

    distance_threshold = 50

    polygon = create_polygon(rotated_rectangle_vertices)
    gdf = create_geodataframe(polygon)

    os.makedirs(output_dir, exist_ok=True)

    all_geotiff_files = []
    for city, city_data in cities.items():
        geotiff_files = process_city(city, city_data, polygon, distance_threshold, output_dir)
        all_geotiff_files.extend(geotiff_files)

    merge_geotiffs(all_geotiff_files, output_dir)

cities = {
    "Seattle": {"files": {f"SEA_{i}_LULC.tif": id for i, id in enumerate([
        "1qVBcR3XkAZfg6egpdJfJcyorkVk2-7Kx", "1EaNzUtFD9pX7FtsTEKDMjwhiLbrUHW06", "1RLRhjkKMlLzIXU184oxz5VFTSLtESfEM",
        "1CRBeZolesa3-Vg8UVVmesS-ZN45wJgLl", "1uwqEKvM4wOaXs43DjwA4pupzOIN-ZHcy", "1NkH3qw424kcYzTgQrsXn8NqfxgmUbjhx",
        "1m8pDcJj-CG6h3ENIVLsPDzk5-m4XZe8h", "1MwacOeCWIjY2THqWexXE8IR6Z4oOkCGg", "1onv4auM28v1QrNA64K9C0uXca_GvTABG",
        "1UVTGOPmimzzYQQv_Ndgp2seUheuuMU-C", "1sVTTRaYqbDRFqw6XrSl9aipx5HZXxPPp", "10sOVlGh1D9lQ5dyFB4fw9CGOCdCqqGnj"
    ], 1)}, "coords": (47.6062, -122.3321)},
    "San Francisco": {"files": {f"SF_{i}_LULC.tif": id for i, id in enumerate([
        "1J7MJNMjpgvUGZLereL7O3cnNzx364cTB", "166YygBXmok1csaavFDSViYBHdR6ZkcXG", "1iJ1E0kAlnpGof7SI9_jLsCRzm1tZYDi8",
        "19Yk1XsXZTa_8J1gpXhFGbHD6HkBy-ZtD", "1DJwDy4O7QOcwjk0oNj6eCUclJAKxI5rw", "1qOvm4lyvIVcN2gxVOEVRmpS2a31yhS2a",
        "19Fps15YqlPG4MQ3S8aOQO37gurKJ4qYL", "16uWJ3lCYhR0iHdIRU2Z3xlwqklv-S2-4", "11NWOOshlfgUQFbwtR98qVTI9Rl-fpqiq"
    ], 1)}, "coords": (37.7749, -122.4194)},
    "Los Angeles": {"files": {f"LA_{i}_LULC.tif": id for i, id in enumerate([
        "1eY4w6UKNiFVYhiRIoA1j3xgd3q2mwaj1", "1H4Ek94o81R9yE-mk2jdw2-bJ1ZwqPn3a", "1w4iM8PhJ1pGlUPmofZAURwtbJ9yUXOYR",
        "18S-8rq6kyK4ZJ3enI79zLrH87LsCTZuS", "1cEKBKFYCunksyTtiWSPVFqwoTXJTGjFO", "1MH4b7Q7diWetAC6Ph9plXeUI9muTD0Qi",
        "1nzjETBuVrWqTScENgV0tnn4vP5O93emq", "13ind_9R3YJGUb2QKU44dmFHmFzMclVnC", "1vJ0w9ppIFdYlynyTpCLiKwdie09BTl8u",
        "113h1iJNG8RloN1X8ktRJBb5nwCmU-UNC", "1U61kpx0w9bMl4mUMnXggNvoFOPIpanAy", "1qtYG-64qgV-1SJDmmLBW7rbO2i4kVJV3",
        "1mIgrfLgYxAPcYLRtZodkqx03394SP4tD", "1nG0uO2OiW3znNmkcjp-bbgS2ig_3aABf", "1dnzSG5Am51z4exO_Y_rs_ZqogdGGW7uR",
        "1vZ2jy6yqdRaVITVcBYc020zPYiO0Eemb", "1vt_xrXStdeIGuKE0yBl87T7HEswLWk_F", "1FItEDIVNIxzGpJz9bUYtd59p5WndirWH",
        "19GZMlwzWIPlxWj6gh7k78KDXifTwZAsC", "1IXHU8lX_JXebuMd6HAKSPrDJZVUxMv29", "1Tj14hrAD66au5rYJiZeq766FLIgcbA6q",
        "1AV9Ii3geoupmogAJbutY5XUbJgrFqYIl", "19n2bQAt7R3tmMOcAjM8ercm_TouV4Ud0", "1ZAT6aNJq6EVflupP2yX3O1p8UH1xM5dQ",
        "13a-lPz384Lg1DWSPxm7r6CCNScJyZAHA", "1bwZHWVPzy0v8_xLX4xZg8eSLjWcwNE-r", "1tQVc791PhJ5mGZIcWaF4rQ6YpP6Wjm8T",
        "1-U_jgDbsKxp5C2MsnV9yKwv10UZYGHfZ", "1LxqapO8fVa607hklk-dO-kCdibbt4YDw", "1DAXlDHPfiz14p9853-RIf-BKLTvmfjS_",
        "1RL8UkrpFP0M34_hu5QAPNik48B_Ejwi4", "1eCLJDZDeZ8Us6oZPF6JxjpJTV6QHbp2V", "11N0OqjK2Hwr9Q2k5aWBSEhpMMdqRmHFy",
        "1jnl3aQ8wRSYCdKID831HZ8TkVeknkhcr", "1lon2yiHou9R7YFqIRFm3aj2V4kH1klHt", "11lSeuntXA--zPK5zVgLuRHEhfGB-VHJ3",
        "192l7tD7eE3-9PcPlNsdYyCQ0viE6fiip", "1XYh9XsxBAAP3WoAmA0DY2HYE-K94Iis5", "10WnLXHrDaJiEHiwWGDtCoB4RMOS9DGn_",
        "1zEW45Po6GBI0IjjWx_UMSpagz3y-dNB4", "1Z_ybI9cDUyxrAW5LonrlyvdzUhXxbfR2", "1fZ1nJxoRGPc4A2rYbfbN2C8mbBIhUtGI",
        "1uQBiX7pIlvvUWG5-c0IOipo2ExSWctud", "1WlU7titcAezUcIH6PILO8ndvMQhOPG3G", "19IJ1SW4W210zjmvJw685DZgL8uBGe5if",
        "1H9OoqiDLbqfyVMaRyX_cKuOHJKDhfnDg", "1UqoaStldyr_tvjzfgg7fib4xSI6wPM1U", "1A5aRQkre-5t2k8gGo-wMf3kuQXGLkeKw",
        "1BuxU4mU41AAXdEH-L9vBfXkz73PtZRnl", "1XYNqRROkD-AhsGksMyRvWnuozIlsCbt0"
    ], 1)}, "coords": (34.0522, -118.2437)},
    "Riverside": {"files": {f"RVRSD_{i}_LULC.tif": id for i, id in enumerate([
        "1MuQNyLdL2vloWaueUhCzXrP5UZfGWUpW", "1AmZ_mH9cTxAipfYMiLwnLNcWLVuztDVB", "1EnobaHsbSdrlAsU6EB7T3aPuzHodqadf",
        "16evkoCTLTWWS8Gc5yY1Eqxcn-FKsnKFa", "1T2mmzn-HXu0_dass-MkOxLO--ghi7qYj", "17DCkIv3xRvYDwfODIEfGZhUX4H5aTAQL",
        "1417-_b8h2f-uLg2k3f_jjiproLsoiQqA", "1PfxWrpyKtF0DJkNC6gtmE8N2HdcvoyB3", "1iPsUUZXcZxMMba5LGxQPMATecj9Flin3",
        "1962DtU1hIz25vJNsT5DxY5_2Fu7NxKOx", "1mS77S7S-OByP7UpxcFvjZWkKIz7-1TYt", "1mekQaJCnmoc2rZ-bUOZnLHsuv058DFy3"
    ], 1)}, "coords": (33.9806, -117.3755)},
    "San Diego": {"files": {f"SD_{i}_LULC.tif": id for i, id in enumerate([
        "1dRnAwkiNI5ASAqdVNnf0v-vs-QWYRLnE", "1YLgbpnq18osmTogobi6e7zhqTmLFVULE", "1PYoZbuUB4cKhhXPotDxFtr0xFursIGjc",
        "1LyYYfjhLa4RO807jLANODsSFkPiyuokC", "1rJCOdP8tmJmWKf_8no_2gtVdePjphPwO", "1LBvcI3hkxvJhodLNh84Ck726qAgKQlfI",
        "1kog0PFgqeL83NYyOe6u6UN1p4Rp2S6lV", "1g6iPAJDIluLjPYKzxEnX4-2wCHReedg_", "1Zvz0u5XNfyqKYuPSLgOMEN0L0H75BE08",
        "1w2tuq4tOfC53sRIQkOcaM-YaXuR5hGvz", "1mptr7aCabd-dvHh35yndcFXM-OkqlD2-", "17WcfGWQd42278r1gPRav_YaDRfe-JQmz",
        "1mNcY8huBOrqGZGcd-zDapfqZhOM5y4dD", "1--tPkVwFtFpUjOkPrFimaWV1uG48JTYD", "1n3SQxF10GimS3VKo5nxuB9JrH3oZQhdx",
        "1Ov29Q8AjtJ_vHzPY-w3gpH1XInvofC6q", "1DbmDRwMzZppXkToxtjbUFsayUPm-yQLg", "13ICoTSp_fyHAmezaNsQ3b6-Q1iysWSNg",
        "1rMnLTuO2DVv9ciHwioPawsuyoidh_b_r", "18meTr1QNnPYrOBSdm9a2JA2pPL5Qh9XI", "1gntUs4qgdPMe8bIX8bmKOnEhkvBvVMmh",
        "1BY0ajN1GhKK8Z_q6_LXojUha4D26QiC5", "12rp3QbmXaZUan9I2mGtQX_5pidRmV0S-", "1mfTVnmUKsXOOkwC1AtJee_DSB4ELE65i",
        "1fPMH8MDD1I5GiRkIpLGhOjZBEnKvjcTL", "15fAxmLwYAJmsGOKnD-3qiMVfnRKf9iI_", "1LoF3-XbZrvA9yZh5J5r2HS7wi1bQTF-u",
        "1mEptsdeX6cR_z9-XJbl6tiab0Yool8EW", "10SP__LRtdApqKEzAlt6_9DGVd7o4olUf", "1VMBTasrTZP7jDapAu3nIdXoG5v2grIth",
        "112DaqBCEl5ETJQb8_VLKB4UBN53WfPed", "1ywVL2Z-FRYqYltVQ_i3S1WbN7RUZlGzu", "1B1fQTQOGsjLwijfrIlWaVYSjlCr5EXB-",
        "1Uo74sF4AxddcnnE0WgMg9aHBQ6nl24j_", "12A_1LAHkmqDfxfi2aVKV6V12jFv4Q0KM", "1hwRvlcj8k1ECWJbgEslzceCt0WmlWfhg",
        "1EedKDrqt-vAktWwsvkMxfOxcCnv81E06", "1N1KhGLFgIQcJhMG2rvw_0jqX-h1jhxC6", "1NyJvD8KtHl1mBNiGMf5zN3mqMaXgiJTr",
        "10P5BebHEfTtp2KlkHSA7hOjwnkzgMyS7", "1e4GBheuIzuD-RtwN1G39XNtNI7fEaVC7"
    ], 1)}, "coords": (32.7157, -117.1611)},
    "Phoenix": {"files": {f"PHX_{i}_LULC.tif": id for i, id in enumerate([
        "1NlK4CBAkKFAfqrb8OkULkZ0ca5kncSg4", "187vUtJuS9GV6Sn1BB1xBvV851EDQwSec", "1Gr-K4uefDyizgb94dGWqNbWh0DXpGuxK",
        "14SrWhJga7Y8p2bKHo0QEj3UaZMNOz6DB", "1P8UIT-4MsTI7l7a0FMjbkuOmqn95wdcY", "126Bbs7e96V7_sOOEjrK3p4H80YMb2178",
        "1jc_R3JpyZR6hGaTdOEwsfUQbWahFu-xs", "1QEeAohyTMZ9bBUkSLsGjC3GB2Lan9qbb", "1gqWj2h-J1nuk23BO1G-3PMmrUcCx31LF",
        "12PPkbOYs6cPmhEbnoq6us0CiDbzU7i6c", "18g7z3LFzqSTtab9lAX9Mv7rScaIOpCHh", "1lcv-qWdsPbhWrNefK33y9vtzNG1ByqjK",
        "1B9lBbsGPnIr7kpMM8pQaouM5-woncis2", "1Bp9-rgITrinGA-UPiKqWNIm1tNAxekeB", "1TC4aGzbb6OP_689JI9yYBsXbbyeXs8SC",
        "1RLpKGFQFz3KGi0jtI8_UszsK-NHE6gNQ", "1XBxU_YysrEk8NsEy6dTfrGp58_8bjkXw", "1EsrXWK5EQFHSL8VxDSi5xH4baFjzkfnd",
        "1H308CG77xUqoSOsMsJIgkKd72z-0lDch", "1DwnNmqDbAkiMmqGlQfBD3QiiBmKVnNMx", "1i0R2EOM7TLhY-HIChqibTKmBq6pPB-bW",
        "1NKW8166hNzbtfO9LiXWlQMQoJu8H966k", "1sM54JEUvF4DjB1XlGSwgao7jFTlwPHa7", "1WntVCYoo2NhBmBvVG-BywIeTp8AejgRh",
        "1SqUme0-L09tYYHbjoMFdw7Pq7Ts2oP7y", "1cHi5kfVA2-cSBcvpWXHXBCnljod0lw_D", "1GwbZ4cmS6076-9low6y-wQhSO4uHmOlF",
        "1461t-k2cZo-w_h_TwbMqgX3Mi4SOLxiQ", "1g3_qAK8CEJ0ynEi0OJ7YBJn8Zs4go6lq", "1lFeC13RvLvaV67eo4tc6PiLg4WFKruVO",
        "1EkpHfvI6EsatB2p-6P4gXoIdIo9o1bPU", "1rTuGpe-b9kfH5PKw8RpTh8a03aEA4_dZ", "1eSceH8H18Q9m3RL1fOD2GdmP4ET9irQH",
        "1KhPM1cTYeRt8u7LmXTyfwWYkCm7qKklP", "1Y0xt_Ddt2-Qfd8iUw_FgwbZq1Mb3xlOE", "1bdQa-ccuOoiXSej0TjkomPaZtidpmX0l",
        "1Yf6EIF8FHFw3weYry90Qc-Z7HrV9k-ad", "1Frhzhi2N3z4b5K5G8qPlMqh-tJb9IAls", "12KPMappaMjDai5iQiRcDgtQp2PJ5IRwx",
        "1VU5eVF847s49dv8Aba9QNCW29V5BguUM", "1PPgcswKhP1ME3kSioh2Vm85Q07uCWi5y", "1epZKlNFcmyxrPt6lxoHkbJF59FgYgojw",
        "1r7Exm_96YCP10mumJ1ddu0kwTOFl2iit", "1tufHH9CUMGCaJXREQPWn4SF4aSSnHfqC", "1MLfPDivNm_VC2op8qQSOo3uCayl99CWl",
        "1OTPLuHql-B6uNYy4FZGLSSYIygwm2MLs", "11O25LNbirEXYlWPwio1WR43FUDX3jN5o", "10gJgaVTwOJXPrnKIunwoAxi7fXm-hMAX",
        "1sSIaxPd1xq6-DWHO0PzaWmP9LSMw2nl-", "1l_SpYNgCQ6j5l5eKiFiQqnFSKzwD1I4A"
    ], 1)}, "coords": (33.4484, -112.0740)},
    "Denver": {"files": {f"DEN_{i}_LULC.tif": id for i, id in enumerate([
        "1cn3prA47tK2XwAk-afmFlVr4cwJnDIdj", "1JRrL8k5w-MpBGHckFuRmbWYJ6FQqtiTL", "1xVLjYqyaIRnoMxI-VERUlEg0O3epU-s3",
        "16a0zTO0Sjqz4k6tQxisV6LNA3TsU0Z-w", "1PgMH9N8Qrwa1J96knQe7xT7rrFXU5kGr", "1ZKt-FRrQJIWYn9wiARmi15hNCt-D8VF5",
        "1dhuR-X1qPsL5fm-C5Cgv1sK2S8kptv2v", "15zfgfrQbujKuayOzM4flOyV7oC6edv9m", "1Apfeu2eo-doVzgJXVPTU-w5TCSmG_3_2",
        "1TB5UiZ1K6k9y9ARsZEh4UC8MC0WRw9_Z", "1JraHt0fAG157YFoEo2IT6AQL-m3JJxp7", "1t9amw47pOYcaypcPN256HDlHl-h0FSzg",
        "1zjz-K9YXtAQmKrwTb7l9HGVzC-Q-hAa5", "1UAqyfihJNkqLn_lkZYlgunPxCmzjE2Kw", "1v8LkrvbzqhGjl87k73nvbxcz1feZF-HQ",
        "1tip8S_cwtotir15rl5wtS4nny9pMwKHm", "15MESgZVdRdyby_5oEcg1O4K-Q0fAi878", "1y6OGgqWW6u62rKtAO_lE0BPsBcunqY7j",
        "1eP9L-2kRYX4_TFUfh0ub9HgIQp4Uh6m8", "1lM21PPBj2oe9HX6hQgnaXQoZqNPBY2jO", "1OHygdCpzDaXnWaievoxbH-NeV7uVjyOn",
        "1kkqVcKTErYzAvr_WZHX_-vKXvw-hXZEM", "1IGYY8mgT2XyMVzEgKN_06yeTFM0jXVa8", "1DOU4YVp4t05dKcJkD8Xlu0rI28ouFiBd"
    ], 1)}, "coords": (39.7392, -104.9903)},
    "Denton": {"files": {f"DEN_{i}_LULC.tif": id for i, id in enumerate([
        "1m-OczszaCHcc6ek30bqzwCgc1p9o7Biq", "1qelKH4RrI68wdpdMX_-x59Rv56pMf7_X", "1l7KUKFLQv2rQVmaCqWtiZqxa8gL9LHIm",
        "19r4tMqAr_jKyHAWsPL5BPG55Ljht6lZJ", "1aPk3HfxFcLzA0sngBSjnkyXym5viI6V5", "118F8S6XM-3n743NyF-wiGYaF21jNjHlq",
        "1n_9wN6jZSkJbu-Hio79Gv6dGZ0MMmlLS", "14e5qHWXHB49NrEmQH9QG1ji1FrB6WDp2", "1WYpfnfJntb12av5zy3SyyPJ1q1BkZkuY",
        "1Iej7mcsW8IoH-Gz8_VWrVhTWQUz1A1V8", "18gcUvk6KZC1rOY4tJXu6WXrLwDwJ-2-s", "18BC72n8WnmbkrYBoSzFagNJHN8vyvmc8",
        "1VqbPnb5b0vvZxaZJWgUCLTMkzOiLaH93", "1yKD0X-xcGm7F5K-wifJzsZ7G8xK1HenV", "11xmcUNH5IaCugaLMhXKHCjvhvqu95EYF",
        "1ny7nOVdURK0fkt94p-63bSoR1_fh8dxu", "1JacUSTM7Lt5Vcle8Nlq-HXCu8FmqmO_1", "1XZcxzf2O37_rdrrlqbAaNiHupkKdXShV",
        "10CfjmI0rViwQVn2aiQ6c_QiN1vK1oaXD", "11GN2rT1IfAWXlZmQe0vEhNb6ld8-w6Xu", "1cO22dkfq2b-gpjFwFOuRK19PlvyuXwBb",
        "18lkKiXV3t9LvDRPqNdpU999cQMNsU6LC", "1jjMagrltRoHFS2540Tm99vtwQ-jZ95IH", "11BBsQKea2OLwDg68NdUo2rd7zmivh9Nq",
        "17cFNa1hNUJgjR0yoOJZoQOJYWZQkADue", "1TWCz-zj39SrY_M6NHbqzdczyiT5XcLIq", "1yWi_SPgtOXou0TrsH0yOSC87hgDXN-mG",
        "1CmMZd-9O6S2DCJtEiyxEB0hwKfFb3EV2", "1CEY4ENB4RzI732Q8dNYqHmVBqMMbYQvd", "16Jeh4r1aHb_gS7-mRuXkoeN87zzHJ5WR",
        "1cnZpSZJsgOf5GpTLAsLBgpdlbfIATP1o", "1nlrf_vQJjmN52Dmoe-WjnmxhhytKFd6B", "1Pu77ju-Crd6zLZ2HQMmZejtrpAMHTuny",
        "1xmaQQtVPfcO8wSI1DYKAB7yqNlkeIXrh", "10jkb_kxdVCdlLeua79XrcU16_-7lS2zR", "1dfaVIT_iqiIHZDyeU7EKXnL0wRymmkMI",
        "1tfegwdAXoFWypPEEwaEZRPK7pvIArZfW", "1FZPFn0_TxIcFZEE6-qmc0KWorv49E76z", "1nxAkOC2YCXXRNS05xhnSHKIsjTdfvSDH",
        "15eU0MjTBIEPktHodtyHj5P3LDmr_UV2p", "1ZnwoVHkWdAURD4BOlFrDymBv_WMHeAnS", "197NQHMwTe81UUgLsIom6xWyjX2iUvbL6",
        "1yHGazeODths0Jyj-qN9wYdnAjgyxoU0i", "1EBUapYcD58fvL-gwOf7MRztio-mgHC0V", "1a8-W6gqEV3d7OFcDTyT_mKhZJ3QG0L8S",
        "1MWwNuZAN5SncWtkhLC8Ic75wc0_Le3jN", "1icfgGyuTbLDT0isgBfNQMdbOKu1G83OX", "1OVTrcTvA7xY1XosgoOtF4yc6Hq4TxL4C",
        "18B_vHEH-BU7Ni3IDa7bAoutmOiRBZXMG", "1cAs5X2Dwv_LhQdlvlsI3oa6nmRC7oYOn"
    ], 1)}, "coords": (33.2148, -97.1331)},
    "Houston": {"files": {f"HOU_{i}_LULC.tif": id for i, id in enumerate([
        "1qI5md0xWAbGN6cPrklFbHYVqezvgzCF5", "1Uuzurxj2Y_vxTPkThzdGHUYq0BCoxOvU", "1_7JNA8t-GAZlLWL80TPDrmyW9kM0dTiE",
        "1wEm8Io-b1EIGiPAhmAD052BKTzuFrwqV", "1IZdh-DUxZ27CugkfBSlL1BW1oaAp8yw8", "1lLK8bZUT7EcVQA9lN4DotcdHfLp229ug",
        "1sjzvs_gW9qZSnOjkB-znEyej1aSzyaY-", "1RxrCiGyDOwSMyVzM9nIWGwHP2EJSiGj-", "1kdPdp2WzMtl75odKTtyUGewEnuJV4ygh",
        "1KKMVzql63y9shh8iVZXkEwexOWdsExiC", "1sow3h3Xwnte_xw__sAFCrDdp66X1ipC4", "1wNuwy--SsrVwPpKIoXyl-BnyRc3oBodG",
        "16e5SOpYzItHrubl2lrgDfMpbhyt3imXD", "1JlZXryEWWsPWA6Wlk4hfUp-ApWT6qLJC", "1vSI8KyT9eRRKJLruDwKudbqfvPGXjLLs",
        "1rLbVW8uo_rKcvPWOsdB2UglZ-Q7wBKkz", "14tUehdQqYZffl3icIcn5T-nWQ5dMQVDg", "1gD743qzUnSFFjEjEGs0x0kdVXNpAjirq",
        "1EvpWHJFr5Ido1-Ip0Mx_d4LSZmL-jVmO", "1sd03uQ-16FoL29FWriFlUrhRA0B6matu", "19bkEvZ9AgcjMY7Ef8DHOV0FKxD4MieZw",
        "17vF1-Lt-9O1a4Cz2J5ZlO3XtvQCVtG2u", "101-W73BEAwnYoa-amCFE4QS-i1P8uEoA", "17Ht0g27tpEkfUYM-zG8i3HW9HTewFAZ5",
        "1_3eiVwGXvIuh3zzJbv-Gh1sLbX9wfPD2", "1m1PmdRlXfp4A9F-m8j-NvVKmzd1_EsBM", "1YiqPBheDXEOig7NaaO_TyYW64erjtkHN",
        "1j3A4CXClQKiQjDEVEoFpSsspkXkbZXf_", "1kx17a7z7sqLkcQ5fyKFpdyJDacgI9xiV", "1eV9i0omfnaDdQW0QqvoSZTRMPY_PkcwK",
        "1w1ZDeCpsaM4FpS7lF0SJY-YLlg8d6gdr", "17DRbRJ9OVl8-2wssIzN2OaKpoU7s2yvO", "1HzxgFvnAHlO8FtYIPQGeYBElwzsLEp1I",
        "1iYNFxgqV4TnJ2bPZPt3bfivNdVwThAAh", "1EskCO-eMM3lwl3te8M04QTt9GQv9MdTb", "1i2LuyDXQPKGn9pgrpqmh4jKbnPKM8uWw",
        "1fg5c77s7tD0hmfpqkQGieFj-Fbde3D9G", "1u9VXfU5Z1vnMUeKOgpL0g4J66QOeQ9Xi", "1c0TQE0f_UYiBqsJB0Q7H1XbdhksXtxLE",
        "1O546lUhwX3tupZ9epy1ioA_1pnV4686k", "1byTgo-9p2BBW4lKjLAMf_kaIZu_8HG_P", "1yuyu0EMLDRX4-8uP50rCFuTUsTrWa6XA",
        "1RFjY7_6oYS1CQZ4J51mcQU-T_m7Fwl_e", "1hZburopaeD6j8eAkT6ydMs0s2uohJLwT", "1jjwwPbYqG0ri40kuObyWxofrtnfdETRH",
        "12GRugaSmwTApzAsukp0QbsOIOE8_ThFS", "1Tio1Uwf7vIuthsdGFkVU9ongGXSnYxqb", "1id_Uvc1oa60ay3A97E0Pl4-_yQkvoLhH",
        "1ZvdVY--xx2TqMbyaJKmKGFzKMapKsuqF", "1m6aqinQG_Oh1dN3bcXpdYLpu9LpyGQ_0"
    ], 1)}, "coords": (29.7604, -95.3698)},
    "Minneapolis": {"files": {f"MINN_{i:02d}_LULC.tif": id for i, id in enumerate([
        "1luKYZcYLxrkKZvL-kOdbt6hOsaLVwVpp", "1Hts67RzAdi9J15EJFaigRUQBOq6hAX7G", "1SfnS6ETL7j5XPypLI3AyrQ6OTwgOcjaO",
        "1N74OFGOdzFRvJuR93CX-SfbA84QRD_wx", "1xDchvAPFqssTaLUljWsGPEk6Uwkp070g", "1GU0QQ3nySYbnFNlWBO8ZnMMH2VW4_pvR"
    ], 1)}, "coords": (44.9778, -93.2650)},
    "Des Moines": {"files": {f"DM_{i}_LULC.tif": id for i, id in enumerate([
        "1_TU2OgShk27YFaQDrj7S_CyRgAChkwzL", "1oy4862vH7b01qutp1CUn7f9vtBAbTOGx", "1nVSYbiz99hm2xKP7yggAKWiL_xwZsbok",
        "1eprOHxBBZrUtn-QW8ZsS11wE0TppqtIQ", "1Lx1cpuXxUUeVoOLPDFiI_Gn_nrzcfO_C", "1mvN3vYlDyTcT6ogER_ZtNh0emsk-hR0k",
        "1xgUSQti3BV0oprac0tz0lSmvekqgGWtl", "1GZ3RQ5ooWfvqSWe5EhOTPp2HS13Si9MQ", "14D3cyg7iqji_EAW4fMjgSkMM_m67S1zm"
    ], 1)}, "coords": (41.5868, -93.6250)},
    "St. Louis": {"files": {f"STL_{i}_LULC.tif": id for i, id in enumerate([
        "14IXaO0Kw42BWulzAvxv6lwkuyK4ONJRM", "1-LnJyI4557NLvaVGS15ftaFZYe1eK1Jl", "13jbmQt1WkC3xc55C3FAi-fN_f6DMlOUX",
        "1DjGYNv5mDCj6TQNxDYmk4o80fLfCq-E_", "10af4mUnWA23rEi_EfX_RuOS8l2b0vL3h", "15VcO08t9bhZ0txNCwc556Gmy1VoBqIoY",
        "1wSy0RKlW1MRYCCC-hALDJFa_OW1Nw7Qu", "1YQVaPbgsMejpNQMGazcwtRjW4b1zpHrt", "1SOzThsvVbaiwBaWa1q65suKcYssjNexj",
        "1t4B4bhJ4KtEpww_F6dsDI4sYXxJ3iurv", "120mSEvITh30tW2Dz9_93wnQX3X2N9rvG"
    ], 1)}, "coords": (38.6270, -90.1994)},
    "Chicago": {"files": {f"CHI_{i}_LULC.tif": id for i, id in enumerate([
        "1UmNuRkZ5t9ymyWnYrv2oAt-34Ajy5W4O", "154qwcuPWsez-YBmlDfAtL_nFHJfF5hJV", "152JouqyQL5tUL48i_61VtrLP-BSszlT3",
        "1H4sbRvvjsHodAgXG8msa6iHiNgJhQcGw", "1vPbH4IJVkgVK_oO0ygx8Dp2oAB2ecQn1", "1fpnzvC0Pq6QCgEqkrgAUDojAW-BpyFII",
        "15dGuxhxixxB-FxaPyyinx-cwxg1gHOCv", "1PphktYyl8jDqvJdhOX5ZyHFTVVLYdZun", "17eKmvTUidc1qwasFpRQtGUItzCMhnavD",
        "1Z37todFhq3kuB-aSzFNmi8XtQTN7HZSM", "1LCuIfpbkRuyVlEuqT3gOSrKUDeKgpuW9", "1hp0gdqo-9Xpju2zFGnOQrarVIV_mndmu",
        "1B45sBkdPl0a1Fa44nobFgFt9mKYyHLDY", "1w2EbSL88aD2Dl8IAkcGM1spTTv8y3Ej6", "1W9MtkMGV2lrdd2p6cK-wyHPKF-tRM6hM",
        "1YCLjTCibJ63Jx4e70lMlE7CRrlpBDZOP", "1vmyirCf7ekjqFdkz3-6lsOPhaLAhMiC5", "1HwyFGrdL_vIKb5Ve0pR4lHN1vVXmuBmN",
        "15LkZH9b6Ng782iCYiaVOOxvSpCKkglGQ", "101CYbUNM68vlpGFNnXGe03hkZtyQuBBa", "1MD1NdagakeB_pcDTiOCRoOfI-ltPhbme",
        "1JPX_bqMGuxiMXKA3jq4r25RziWy_q1ma", "1g1_zFG6oUq8-_KnuiBHkSAxRndzmPo_Z", "1ZORTz4H9tw_DwtHyVFgx64RhrNoJuLEv",
        "1piUdHHl44UftWEYtEyPvqN-Yo1WNnlFK", "107RR2k1yd4IGCToo4NTE4Lc2DTnPdB2T", "1gXVGeQENPhZTGb7t8GBV3DicnEA1ESy4",
        "1jkdZWtdyTA0daSTxAh86hYMxzufv1qfo"
    ], 1)}, "coords": (41.8781, -87.6298)},
    "Atlanta": {"files": {f"ATL_{i}_LULC.tif": id for i, id in enumerate([
        "1nPgm8rEjYgXBYzDHH5pJqSoNiepfKlVH", "1Zq5de8VcM27Ben_Y3eJYEs0O1lKg780c", "1yOAuvDNHApfe13HrdnqV0EOUuq9gf4P0",
        "1cyrfHwaQxg_ZoOE_aFAHZUDUE3S01eaS", "1IMKbDIXTU1TDj4t4wuUTfBtnO6tyz3Et", "1Qgox_c_b6Qo0qbYapTBvT7kE1WRLzzlm",
        "1v52vGJeiTDvLbPwV4o-hNkTXhmhE6Wok", "1K6QaDzHs9WTWtvXwpNrGNDy3E9tMVd3f", "1PXrMYw-r0GZ8cEV83fqOuiRvyk8PIHQO",
        "1Yv64k8MGJj1Fv9mhfnZqj5ggCqhh0Hnj", "1JuoCqQko3amEbRC_do9yIiNrjaZuLt7H", "11gwy5JdhvEuuoiwWDHziJqYNLUhooohq",
        "1iztk-B00Q4nbvwrUwPecu-touTmo1F0T", "1VxcyfG4sc5bmkxXAlMRL--PiSTAiNdwy", "1qmm9JejgD2BBSWM2jm55rXb2v73yMuTQ",
        "1_lk2vnyL7tFqEepY77Dz9kM_o-xgWF_F"
    ], 1)}, "coords": (33.7490, -84.3880)},
    "Detroit": {"files": {f"DET_{i}_LULC.tif": id for i, id in enumerate([
        "1oxYi3gZnH_svpz-1Ng2Y_426W0t9LjfW", "11-Q-jQ9Ily9QSH6UJ1ZTg6_hLKzy8Vmw", "14lCrhj5gKgZxpdrH7us2wrvVvaLhtGFU",
        "1WV5HKZSGLMtZfjsWnYr2YTpccV5XwxbE", "1TQgmC7aMZkd4iHefo3cXKKKkiN4g2MG0", "1DfiawSdkAJCAyXBCC3kZ1Kk7EJLhOdZN",
        "1awfoQk4LqU4jKqzABpAQNrxAteHWOvtx", "1qhmEoZBTGY2w6e2DE0mBNOiWC7arqkip", "1IsUnEE-cbN8Jofv1RScAh9jZKJB0G118",
        "1wYd4t6FUAE8N38CHUtVdLrCkcgBfn1WL", "1ZOuAsoRXh_4sR4T5L5JTIX8_TOk5iVWw", "1Iuk0VdKr8Q4cZAp_hjEZh6oX-2b8yv2S",
        "110BXXxNXzi75xCf3vyF73yzFL24MfoIa", "1A_-ck2pm7guPlPY39yPTgH9PG59aL_LH", "1mXnTLUSLeR77u8HVR5YTGWH92HIV9PxL"
    ], 1)}, "coords": (42.3314, -83.0458)},
    "Tampa": {"files": {f"TAM_{i}_LULC.tif": id for i, id in enumerate([
        "1fcTnle-075gIGX_5i_69WJH6jHSJ13dL", "1B5tE5Y0yjdwgkHjm1Ml2p9NGFTUqvIax", "1IMQd5Bot2KtcNq3Hl8qMhwdtE_DCkCca",
        "1yfBLrZrYF0ZAE--8RS0xLO53CmDWs3yn", "1yriH6nS-meVNB3X-LFl5vahpBhUFyHj_", "1QPicTKKaZiKLZLp2uLVXiQytM0NDhLrA",
        "1_4tOB3PNZyDqUVFzVCX6y8feYA0fc9IY", "1P1j6_P0Q6IzM1BmYkEF33hm7I7WSzOGu", "1NcDqoMfYUBWL3jIT8HC6IrdxoJGC-EM7",
        "1cZhWoMa35PWj4pNHMTJebpz8U3knFi19", "16l3u-G5dHYNUOgR24JzLK4i0k7YJs_N9", "1Cvw8PEadCDbaBgmGIIXAUTAQsKGC6Plx",
        "1Uw9eJkkgdp6R-XU3W1bWF2Qtxp_GVZxV", "1hwwCXS5mT6Y3B6wKua6MtYpfmAi7tOK-", "1WTPLppcLw3-AsSIPOe1uevuL7gkJAegy",
        "1bnv19leG-pAPh6BYY8VjE_B6tSIu6WfC", "1to5UMpveHRkrXCVR0csF6NzxWhP_onZI", "1gDH3-HesZdQcEl2iZkqVhf01ktO0H1Af",
        "1lCAPCVxu4swkTElO6jQ0N00r6XUNhdJj", "1Jmft0QGImgl5J7ws-2EshfOk6T0lW3_G", "1LkXb0hpEaP06wj50V-OHEHLJdWFhYISb",
        "1nXE1N_rasmXHWjKxdTWk_SWveCQ75O4R"
    ], 1)}, "coords": (27.9506, -82.4572)},
    "Miami": {"files": {f"MIA_{i}_LULC.tif": id for i, id in enumerate([
        "1u4RGOY26zrxjA6ljORLoD5Blz72qhVWJ", "1-93P0HYr1b1rB3PWOXDduzUTQoIJrFF_", "1_OJuN-TjfCWjCIAPNdkqkNIa595HDIRN",
        "1487H4Z_N0A_UDPYJDrB5v1GDNFBZu3Sr", "113TLAteCDshmZA4JSzgbbycd0tTlIkad", "1R0xqvtzZl1YAvvzL46Zkc16y7lfi_fYY",
        "1Fpnbnqyd6_F9SFus3KOoyrmOMstP1bN8", "13ONi8WXU1Hmw_u8d7_w_CiAsm74rrI_w"
    ], 1)}, "coords": (25.7617, -80.1918)},
    "Charlotte": {"files": {f"CLT_{i}_LULC.tif": id for i, id in enumerate([
        "1_1RELNCy11UAyLDMvX0jC_nbAqnGexB6", "1wFKVAejh5u1zqVhbXy7iruqtRiX8IuO7", "1MHP8Ezs3iBtRKp--FKGNLAQpHiiLZu0w",
        "1e9ud78EQc4nlTYdw8UNFDv8guoNJHewf", "1pOErt19EEUR9y8vfdbW5SNDHEzNPPlep", "1DAraW0CTp_AUTeCCdSEQNSlf50NytnAm",
        "19W4JY5ESobmihcSfy3QJZsLxsyGKi7TT", "1XE0BvOTxV1sxr1PlVxHxxug3Nq1OHY3Q", "10ujdapYhjzDD8sneuv6fgx2mo1-8MrGc",
        "1j2wgn3eHIqa3rBMuCRaveGGnLK7CSlZS", "1VTxpNJY8MJKxDT9HehtrUVMk8e8wX6Ee", "1_r-B9p47fCft_g8mhf9w4IwJy-z4L57O",
        "1qaaZsyCLAD1Ej152iJchdOAPkajS1WUZ", "1m2YlwrSqYcj_KP9v9dAsfj_SbwTWUW9D", "12SRFh5zl_KBnM0VH604IzkzijDcMQbPo",
        "1m_FosPa7u77EfEkLFtyBnuMxEZs0tSk8", "13JdfGTBliBQqZIzXyLj9w9R0WXvZKlUx", "198dmKCs2wSJV4nVij67xU7VR6WQ1beTK",
        "1Vco-h7pWCAFmWZ50LflYSF6SbXBMR-88", "1tQPh37zh6UqIBM3MwLQd_y5GXvTOEIqI", "1_x3f42FynGdR-6R2E8X0VIUi-fqalQUI",
        "1vwsO19PexSiRIeSzWwb7B3rquYAjdWd4", "1xYM0vJVHoxGR_Ypw3DW2i4cn_g_Y35TY", "1Z73xR1pNetuWxB2rVAErVWN9-pffxB2I",
        "1PRev-33rb5WMrtGTpCc6JjgwtYwrOmj9", "1IuDl5V-QwMAUjSG7VeMOg2UhZxrzY8nr", "1LTp_UkAj2cQ1IsCEnAJC91f5ZfumqACr",
        "1Eu9wkaUw9B_77DXMrB7SQ3NvhW85Px5y", "1-ZkuqEc_jfcu34B0a4onHGfegx44ZPdM", "1GOaDKkeb54c_NwNc_4C383dJmnzNopkt",
        "1I9jlWdZtvvkLZ-ATgfuVujfG5TDGyVdL", "1jFP0KZ5CwOIEBkGNBWadDD0uCUDE8LET", "1u9f_Kj7HcKfxrVCjGEVkdOXv341F3n76",
        "1Kk2d0cW3SP0SorsFNjfULKtilctqMMgY"
    ], 1)}, "coords": (35.2271, -80.8431)},
    "Raleigh": {"files": {f"RAL_{i}_LULC.tif": id for i, id in enumerate([
        "1l1DKjMRxVUQPIu5MQLnxiUrKZV7diITg", "1GpxAoFDIaSGoQ0R0mStOfI8YNLBh87fJ", "1wikKeaiKjtmbjy5C_GE5Yc9RDJTrgwKo",
        "15ZUPZObuPZN9ypSz5Mk88-3KOmZXOiTu", "17EaCqa0bjQPMX9HrG867VZzcOI54ieSP", "1OhdSY5lZmdMAEA-AOhHpsR4u9EcDmlSB",
        "1sop8YUGJ0zidM3PKIaroiZfohy_5_Xg5", "1UzyilEEeyVrNZIu5L3RrkpUMaB5dt5_n", "1iofhm83Sls_BR0wRrZWV21ZmXIoKN-Dk",
        "1eWb2wwZdHO4vBVFP3TkoF_yVtOcbxk3G", "1fPmSsILfoxkeuUAD4X02pBIRz6C5-u0F", "1P-_kPvfHE8a_4Km3ryg6rDrYe9fmHdbz",
        "18PFZdPeLpv1P3LJxwIzYNA3l_5yS2MES", "1dRjN9rEwREhvtP8mQAie-4boSSCtBHvd", "1zpsyfXrH2bdDcgAWAFEK4Shpuf1XBkkt",
        "1-xe6pqZn43TvyI0GKN5N77OXtgoOm-FQ", "1bGm7giLLnwL4hTgvPGptCx2YKJtYyvon", "1hJ0uWOYi_68P4qssyIsBGpBXlE2ig6Bf",
        "1TV1aX4EJEnZRRcuPh_ohmA1dcq-aVL5R", "1_FoJcN5l_r9dGS-a9tg-4bwAy1X23SKq", "1ENgJxr6NmfRYVp_8TWMszVHCOsV-kEWV",
        "1H_xfzcXxUzjsIqqsUWXHxlUMtxnL8r7W", "19AnYU0KpwtzvzbJ54tkitvuVo2no1WVZ", "1DYAcX0KpVhIAIVsEZU7mU4k56sJlRkwz",
        "1Mn1SsoWLI0lveZ_dlyfYtFmJVZTSEV1a"
    ], 1)}, "coords": (35.7796, -78.6382)},
    "Washington D.C.": {"files": {f"DC_{i}_LULC.tif": id for i, id in enumerate([
        "1Q_9NCAsjhMYKxiPlCjFhqUTgbCu9ggzP", "14LmaXpl_aoVIyyItThTxs018Az4OEfWD", "1nPbyRtbadd20UEMujLFdWNP2Wqwpf0Ol",
        "15oiWeHDGUNH0HZEZ3kcwNbcNJe3CSanz", "1y82Ar-L4nktrJcbuXxy5sgnWt4Xkyb2C", "1zt8dLXXdI_6zzAdN3cVld3yFIo6iESqP",
        "16D-PRTyna1AsTQUIjXXDKmyWOz1Ksgdi", "1o-99rym_9oJVQCGveFGyDZOEys4zJu0E", "1Q9tzd50oMCc8UdPK4LJCuIDiqhB7cfJ-",
        "1dHsiIOGY1jLbiRxdg-nXZ0gy5wf0UAFv"
    ], 1)}, "coords": (38.9072, -77.0369)},
    "Philadelphia": {"files": {f"PHI_{i}_LULC.tif": id for i, id in enumerate([
        "12Cv3j6cfABkVH1qpzi937zgjN-eRR7Uo", "13xPydeYwx-KIOoJCrp9cWzkNMjkcO0kf", "13cGqj8WCcioduriYvW4CQW7WscZX_JX5",
        "1TnZpKz_-1qh90OQMLw16O4N7JMo22x6C", "1h042Hemg5KTk4qJha917FyOaB0_eY3uJ", "1fyHYTDgm7HfmWoF1b4_CsZQGj8KN_g8_",
        "1M5k9IApO7tuWslCJqLqKQabjrQanLL_3", "1mcK3idagyxFVKT741NBO1skwtjR_yu0e", "1yTrbatWW34XPY-_hMBvYROO-2FBFM3O7",
        "1_RDi0nMz1BnNw4IN9aYCq82DXafe2MHd", "15JbL64QsdYigHtaL0wqdmAPvMsIQ9V-e", "1nt7UNMsl4TonBNhX_oOR286PrP7ociJ5",
        "1Uq3c_EngQuiXDEaMgtd6lA7QUhPX3ul-", "1SbzgvayyaUKjLclIeLz5WK1B5IvaR9aS", "17SEBH-IEziIL6oxcu7xC8_1ws8ZQNFI9",
        "1PoEuTQ-I45FdUwVilTIrNM1KtZCUNj_J", "1cLIDeUwz-su2d4wCJe2xEXUTpIEp3qw7", "167RvhT7Z7CEGABp38rS84mQjkndyX69_"
    ], 1)}, "coords": (39.9526, -75.1652)},
    "New York City": {"files": {f"NY_{i}_LULC.tif": id for i, id in enumerate([
        "1I_Hj0Qi7GsRvxu-Wj0O9nLIcXJ9BqejF", "1xOMSqupcYM9mU6o47TXLYtfgomcIWbX9", "1i319nQw_zWZHiP6MAMCEQENGEIichKHc",
        "1TGUFRUKJ0RRgwVvJcnUG4YrkiHkgRApk", "1HUeCIVTO5Uh4g9gzlSTvtzrgPBSerJJ2", "1od0q5uFxPQsj3dPhNQ5bq2fGA9rDZiAq",
        "1oYyu86j0RWGi5IA7r9kgp4cm61trApCp", "1skfWPPw2rRZtr5OQ2E9vGwC-OG-qEz9G", "1t2nO6mw0GpDmOV9bym79PZWrum-IUCWy",
        "1ytNaDWVyL9mwPCKtZnZChjk44t31Dy22", "15QiGNk9d2t_bq50vy4tQ41ycaQvXn4cq", "1A8Dw06YNisItzSiXKLvVYGznwftKGMQJ",
        "1ndfAOpwZbzEhQtUO-OdsYxBKSB6-O_I-", "1_tnx-AqCClRFP2j0qb3iArk2hrxg83_G", "1DswMAkM5oWHbcSoLBMph9P2Q4fKTATQH",
        "1ac9Be6b82o3czF48Zpclgt67sOV8PJD6", "1foYIhdY44GJLUa4aiuc12fiV8tacjfp1", "1o-SEPlFNEHubiuEwOtVbY5YeRAB3A_hH",
        "1mvSDsOQmulV8deHg72yTivgCKfQr9TK3", "1BGpBJeuA2UeM5bvfTJbIq8iLa_7YUBGc", "1RGLDwUzL122_cWdQqIuJqEnjPlwTyUuc",
        "1jrd8CVg0uAvQzxJ2FDrgSch3QWBA3xgu", "1lhVOLViXTmBLJSwkp1MHgAJYnlNFD9Nj", "1PDGTWzbo--8GTn5RhAtErEsSD0-apBeL",
        "1-wSSrc4OP9JeDnpxa8h2TEqf43QaCIhL", "1oakcY6eKSNEfVfMAAV5bkxMFNLPP2ZKt", "14250P2FksDfOqzuBksJmWPN_gRFWPmAv",
        "1sd68zCfKow7P3C1tPr7xvypOAS9w8Bbn", "1viGPxkPcn883Az-oWi96-forereD5tls", "1fZXCQtyy9Skxde3Vm2EWRfMi0j3q2M4t",
        "1cVQYMoNS0QY5qdLwRm8EBm13KL8TwRUw", "1bK_ijToQgjWF6fv5hKL714bkCTajnrci", "1DcD19xjp3d4zqpKFEnasXSuB7jphebGl",
        "1s2bGFOjZghrwAGf-8iVoSL7JzN0-BaBf"
    ], 1)}, "coords": (40.7128, -74.0060)},
    "Boston": {"files": {f"BOS_{i}_LULC.tif": id for i, id in enumerate([
        "1oaSCDJmCOlCTV632zRIDn-xycurRpC7q", "1ThOE566FbCl5fhzSg-qdNjRdjFxedyB6", "1Q8svVwtQ7VhVTY_D846Hip4aiBCiGuZY",
        "1h3Iad5OcfQ1oVsHyYNEX1fWr8C6b5SUx", "1pt_nsJLa_26aGFfAYkpMP_leLStEBBN_", "1dS--K5f6zBFg_lOPC7yTaZfgT6kYkGgR",
        "1CJ4LK_0EhSXahQWrvOg8mm37Xnm-vGln", "1lTycF0_E9IwZkf5EW7eNpSn_UhftCrTU", "1o5THzYh8SJmT3RZzWx-qEHMR3DvAM_ZT",
        "1OhKhd39S8GAzhmEYiKmeqqjIRQMjVGku"
    ], 1)}, "coords": (42.3601, -71.0589)}
}