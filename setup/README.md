# How to setup environment

以下のファイル構成を作成
```
prj-dir
- share
```
リポジトリをクローン
```
cd prj-dir
git clone https://github.com/acesulfamK/AlphaSymbol.git
```
setup1.shをshareに入れ、
docker_container.shを実行
``
cp ./AlphaSymbol/setup/setup1.sh ./share
./AlphaSymbol/setup/docker_container.sh
``

VSCodeを docker container suragnair_no_port にアタッチ。
コンテナ内にワーキングディレクトリを作ってsetup1.shを実行。
git clone。

```
container$ cd ~
container$ /share/setup1.sh
container$ mkdir work && cd work
container$ git clone https://github.com/acesulfamK/AlphaSymbol.git
```

setup2.sh, setup3.shをAlphaSymbolで実行
```
container$ cd AlphaSymbol
container$ ./setup/setup2.sh
container$ ./setup/setup3.sh
```