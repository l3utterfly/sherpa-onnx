#!/usr/bin/env bash

set -ex

sed -i.bak 's/1\.11\.5/1\.12\.0/g' ./build-ios-shared.sh
sed -i.bak 's/1\.11\.5/1\.12\.0/g' ./pom.xml
sed -i.bak 's/1\.11\.5/1\.12\.0/g' ./jitpack.yml
sed -i.bak 's/1\.11\.5/1\.12\.0/g' ./android/SherpaOnnxAar/README.md

find android -name build.gradle -type f -exec sed -i.bak 's/sherpa-onnx:v1\.11\.5/sherpa-onnx:v1\.12\.0/g' {} \;

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find nodejs-examples -name package.json -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;

find mfc-examples -name "README.md" -type f -exec sed -i.bak 's/1\.11\.5/1\.12\.0/g' {} \;

find . -name "*.bak" -exec rm {} \;
