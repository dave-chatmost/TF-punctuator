#!/bin/env python
import re,sys
if len(sys.argv) > 1:
    print >>sys.stderr, "%s < in > out"%(__file__)
    sys.exit()

for l in sys.stdin:
    nl = re.sub(u'\x1c', '', l)
    nl = re.sub(u'\x1d', '', nl)
    nl = re.sub(u'\x1e', '', nl)
    nl = re.sub(u'\x1f', '', nl)
    nl = nl.replace('  ', ' ')
    print nl.rstrip()

