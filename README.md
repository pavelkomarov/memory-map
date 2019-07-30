[![Build Status](http://img.shields.io/travis/pavelkomarov/memory-map.svg?style=flat)](https://travis-ci.org/pavelkomarov/memory-map)

[Numpy's memmap class](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html) is fantastic for storing and handling big data parsimoniously and efficiently. [Read this article](https://manybutfinite.com/post/page-cache-the-affair-between-memory-and-files/) to get a better appreciation for what is going on in the background.

However, raw memmaps from numpy do not contain information about the type of data they contain nor the shapes they were at write-time, so reads are ambiguous unless you've somehow remembered this information across sessions. The MemoryMap class solves this problem by including this extra information in a file header.
