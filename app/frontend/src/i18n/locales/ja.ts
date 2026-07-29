import type { Messages } from './en';

// Typed against `en` — the build fails if any key is missing or mistyped.
export const ja: Messages = {
  common: {
    close: '閉じる',
    newSession: '新規セッション',
    openSession: 'セッションを開く...',
    getStarted: 'はじめに',
    dontShowAgain: '次回から表示しない',
  },
  language: {
    label: '言語',
    english: 'EN',
    japanese: '日本語',
  },
  nav: {
    area: '対象エリア',
    generation: '生成',
    edit: '編集',
    import: 'インポート',
    zoning: 'ゾーン',
    solar: '日射',
    view: '可視性',
    landmark: 'ランドマーク',
    export: 'ファイル',
  },
  splash: {
    title: 'VoxCity へようこそ',
    subtitle: '新しい都市モデルを作成するか、保存済みのセッションを開きます。',
    loadFailed: 'セッションの読み込みに失敗しました。',
  },
  guided: {
    setAreaTitle: 'まず対象エリアを設定してください',
    setAreaBody: '「対象エリア」タブで都市エリアを選択してからモデルを生成してください。',
    modelRequiredTitle: 'まずモデルを生成してください',
    modelRequiredBody: 'このワークフローを使う前に「生成」タブで VoxCity モデルを作成してください。',
    actionSetRectangle: '矩形を設定',
    actionLoadingMap: '地図を読み込み中...',
    actionLoadMap: '地図を読み込む',
    actionGenerating: '生成中...',
    actionGenerate: 'VoxCity モデルを生成',
    actionRunning: '実行中...',
    actionRunSimulation: 'シミュレーションを実行',
    actionExporting: 'エクスポート中...',
    actionExportCityles: 'CityLES をエクスポート',
    actionExportGeotiff: 'GeoTIFF をエクスポート',
    actionExportObj: 'OBJ をエクスポート',
  },
  previewNotice: {
    heading: '3D プレビューは無効です',
    bodyWithDims: 'グリッド（{dims}）がプレビュー上限の {cells} セルを超えています。生成・編集・シミュレーション結果・エクスポートは引き続き利用できます。',
    bodyNoDims: 'このグリッドはプレビュー上限の {cells} セルを超えています。生成・編集・シミュレーション結果・エクスポートは引き続き利用できます。',
  },
  voxelVisibility: {
    heading: '表示設定',
    hideClasses: '要素クラスを非表示',
  },
};
