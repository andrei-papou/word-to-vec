#!/usr/bin/env bash

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xvf simple-examples.tgz
mv simple-examples/data .data
rm -rf simple-examples/
rm simple-examples.tgz