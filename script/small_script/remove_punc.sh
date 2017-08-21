#!/usr/bin/bash

# Created on 2017-07-07
# Author: Kaituo Xu (Sogou)
# Function: Remove £¬¡££¿£¡ in file.

file=$1

sed -e 's/£¬//g' -e 's/¡£//g' -e 's/£¿//g' -e 's/£¡//g' $file > ${file}_nopunc
