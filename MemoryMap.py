import numpy
import os
from struct import pack, unpack

## @author Louis Newstrom, Pavel Komarov
# Numpy's memmap class is fantastic for storing and handling big data parsimoniously and efficiently.
# https://manybutfinite.com/post/page-cache-the-affair-between-memory-and-files/
#
# But raw memmaps from numpy do not contain information about the type of data they contain nor the
# shapes they were at write-time, which makes reads ambiguous. The MemoryMap class solves this problem
# by writing memmaps to a file after a header designed to encode this extra information.
#
# The file contains the following data:
#	8 byte binary integer containing the number of arrays
#	for each array:
#		8 byte string containing the numpy data type
#		8 byte binary integer containing the number of dimensions
#		for each dimension:
#			8 byte binary integer containing the length of a dimension
#	for each array:
#		binary storage of numpy array as memmap
class MemoryMap:

	## The format symbol used by `pack` and `unpack` to convert shape integers to binary format
	PACK_FORMAT = 'Q' # means unsigned long long, gets 8 bytes (allows *really* long arrays)

	## The number of bytes that types and binary shape integers takes in the header.
	PACK_LENGTH = 8

	## Static `create` writes a MemoryMap header to a file and does nothing else.
	# @param fileName A string containing the path to the memmap file.
	# @param dtypes A list of strings containing the type of each numpy array stored in the file.
	# @param shapes A list of tuples containing the shape of each numpy array stored in the file.
	@staticmethod
	def create(fileName, dtypes, shapes):
		if len(dtypes) != len(shapes):
			raise ValueError('The number of types ({}) does not match the number of shapes ({}).'.format(
				len(dtypes), len(shapes)))
		if os.path.isfile(fileName): # if the file already exists
			os.remove(fileName) # then repeal and replace

		path = os.path.split(fileName)[0] # optional subdirectory in the fileName
		if path != '' and not os.path.isdir(path): # If it's specified but doesn't yet exist,
			os.mkdir(path) # then create that subdirectory.

		with open(fileName, 'wb') as outfile: # write the header
			outfile.write(pack(MemoryMap.PACK_FORMAT, len(dtypes))) # write the number of arrays
			for i in range(len(dtypes)): # for each array
				outfile.write(numpy.dtype(dtypes[i]).str.ljust(MemoryMap.PACK_LENGTH).encode()) # ljust pads with spaces
				outfile.write(pack(MemoryMap.PACK_FORMAT, len(shapes[i]))) # write the number of dimensions
				for d in range(len(shapes[i])): # for each dimension of the array
					outfile.write(pack(MemoryMap.PACK_FORMAT, shapes[i][d]))

	## Static `open` reads a MemoryMap file's header and uses that information to declare references
	# to numpy.memmap objects stacked in to the file at calculable locations. These references are
	# lightweight: No data is read from disk until a user asks for a piece of one of the arrays
	# referenced. In the event that chunk of data is not in memory, a Page Fault is triggered,
	# prompting the system to go find the relevant chunk of the file on disk and copy it in to the
	# Page Cache. Memmaped files then point to those pages in the Page Cache, whereas other kinds of
	# file reads demand an additional copy to be made in to their process' memory space.
	# @param fileName A string containing the path to the memmap file.
	# @param mode A string containing the mode in which to open the file: {'readwrite', 'readonly'}
	# @return A list of numpy memmap objects.
	@staticmethod
	def open(fileName, mode='readwrite'):
		if not os.path.isfile(fileName):
			raise ValueError('File "' + fileName + '" does not exist.')

		with open(fileName, 'rb') as infile:
			n = unpack(MemoryMap.PACK_FORMAT, infile.read(MemoryMap.PACK_LENGTH))[0] # num arrays
			dtypes = []
			shapes = []
			for i in range(n): # for each array
				dtypes.append(infile.read(MemoryMap.PACK_LENGTH).strip())
				s = unpack(MemoryMap.PACK_FORMAT, infile.read(MemoryMap.PACK_LENGTH))[0] # num dimensions
				shape = [0]*s
				for d in range(s): # for each dimension
					shape[d] = unpack(MemoryMap.PACK_FORMAT, infile.read(MemoryMap.PACK_LENGTH))[0]
				shapes.append(tuple(shape))

			# calculate offsets of the arrays
			offsets = [infile.tell()] # where the file pointer is after reading the header
			for i in range(n):
				# next offset = previous offset + length of each number * number of numbers
				# product of empty array should be 0, but numpy returns "the neutral element" 1
				product = int(numpy.product(shapes[i])) if shapes[i] else 0
				offsets.append(offsets[-1] + int(dtypes[i][2:])*product)

		return [numpy.memmap(fileName, dtype=dtypes[i], shape=shapes[i], mode=mode,
			offset=offsets[i]) for i in range(n)]
