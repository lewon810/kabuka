---
trigger: always_on
---

## Rule
- チャットには日本語で応答してください。
- 原則コメントも日本語で書いてください。
- python実行時はvenv上で実行することを忘れないでください。
- venvが無い場合は必ず初期化し、requirements.txtに必要モジュールを書いてください。
- 実行しているOSを確認し、Windowsの場合はPowerShellを利用してください。
- PowerShellで出力したファイルの文字コードは常にUTF-8になるように注意してください。
- コードに修正を加える際、README.mdの更新が必要であれば日本語で更新してください。
- 学習データを増やした際は全ての予想・学習プログラムで特徴量が一致するようにしてください。
- ディレクトリやファイル名を変更した場合はREADME更新とtestがPASSすることを確認してください。
- コードに修正を加える際、他のソースコードの動作を保証するようにtestを作成・実行・更新してください。


## あなたの目的
- 以下のURLに記載されたデモコードをベースに株価予測を行うプログラムを作成します
- https://www.kaggle.com/code/ikeppyo/jpx-lightgbm-demo

## CI/CD
GithubActionsで実行します
- predict-scheduled.yml
  - 毎日８時に日経２２５の採用銘柄をYahooファイナンスから取得します
  - １週間後の株価を予想し、結果をhtml形式で出力しアーティファクト保存します
  - 上記アーティファクトをGithubPagesとしてデプロイします（Pages設定でGithubActionsを設定済み）

## 基本プログラム
- demo.py
- 冒頭のコメントに記載しているkaggleのURLをよく読みアルゴリズムを理解してください
