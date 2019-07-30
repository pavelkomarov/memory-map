import unittest
from MemoryMap import MemoryMap
import numpy
from shutil import rmtree
from os import mkdir

## @author Pavel Komarov
# Ensures the MemoryMap static methods work properly
# The unittest framework spawns processes to run each method that begins with 'test' in its own
# TestCase object, separate from other tests. This enforces that tests are independent, so they can
# not rely on outputs of other tests.
class TestMemoryMap(unittest.TestCase):

	## This method is run before any test begins.
	@classmethod
	def setUpClass(cls):
		mkdir('Test')

	## This method is called after all tests have finished.
	@classmethod
	def tearDownClass(cls):
		rmtree('Test')

	## A test to ensure memmaps are created with the proper header
	def test_create(self):
		MemoryMap.create('Test/test_create.memmap', dtypes=[numpy.bool, numpy.float32, numpy.int],
			shapes=[(30, 15), (8, 16, 32), (5,)])
		# Now read the raw bits from disk to ensure the right header was written
		with open('Test/test_create.memmap') as f:
			written = f.read() # independent of MemoryMap.open
		knownbinary = ('\x03\x00\x00\x00\x00\x00\x00\x00' + # little-endian 3 because three arrays
			'|b1     ' + # boolean type. | means byte order doesn't matter
			'\x02\x00\x00\x00\x00\x00\x00\x00' + # two dimensions 
			'\x1e\x00\x00\x00\x00\x00\x00\x00' + # 30
			'\x0f\x00\x00\x00\x00\x00\x00\x00' + # 15
			'<f4     ' + # little endian float in four bytes
			'\x03\x00\x00\x00\x00\x00\x00\x00' + # three dimensions
			'\x08\x00\x00\x00\x00\x00\x00\x00' + # 8
			'\x10\x00\x00\x00\x00\x00\x00\x00' + # 16
			' \x00\x00\x00\x00\x00\x00\x00' + # 32 is the space in ascii
			'<i8     ' + # little endian int in eight bytes (default numpy ints are long)
			'\x01\x00\x00\x00\x00\x00\x00\x00' + # one dimension
			'\x05\x00\x00\x00\x00\x00\x00\x00') # 5
		self.assertEqual(written, knownbinary)

	## A test to ensure the proper exception is thrown if creation fails due to wrong input
	def test_create_except(self):
		self.assertRaises(ValueError, MemoryMap.create, 'Test/wut.memmap',
			[numpy.float32, numpy.int], [(100, 20)])

	## A test to read an existing .memmap file
	def test_open(self):
		knownbinary = ('\x02\x00\x00\x00\x00\x00\x00\x00' + # two arrays
			'<f4     ' + # first is float32
			'\x02\x00\x00\x00\x00\x00\x00\x00' + # two dimensions in the first array
			'd\x00\x00\x00\x00\x00\x00\x00' + # 100
			'\x14\x00\x00\x00\x00\x00\x00\x00' + # 20
			'<i8     ' + # second array is int64
			'\x03\x00\x00\x00\x00\x00\x00\x00' + # three dimensions
			'\x19\x00\x00\x00\x00\x00\x00\x00' + # 25
			'\x19\x00\x00\x00\x00\x00\x00\x00' + # 25
			'(\x00\x00\x00\x00\x00\x00\x00') # 40
		with open('Test/test_open.memmap', 'w') as f:
			f.write(knownbinary) # write directly, avoiding MemoryMap.create
		refs = MemoryMap.open('Test/test_open.memmap')
		self.assertEqual(refs[0].shape, (100, 20))
		self.assertEqual(refs[0].dtype, numpy.float32)
		self.assertEqual(refs[1].shape, (25, 25, 40))
		self.assertEqual(refs[1].dtype, numpy.int)

	## A test to try to read a memmap that doesn't exist
	def test_read_nonexistent(self):
		self.assertRaises(ValueError, MemoryMap.open, 'Test/lol_not_here.memmap')

	## A test to ensure readwrite is passed down to numpy.memmap properly
	def test_readwrite(self):
		refs = MemoryMap.open('Test/test_open.memmap') # default
		for i in range(len(refs)):
			self.assertEqual(refs[i].mode, 'r+')
		refs = MemoryMap.open('Test/test_open.memmap', mode='r') # pass some other mode
		for i in range(len(refs)):
			self.assertEqual(refs[i].mode, 'r')

	## A test to be extra sure offsets are calculated properly, including for empty tables
	def test_offsets(self):
		dtypes = [numpy.uint8, numpy.int16, numpy.float32, numpy.dtype('|S50'), numpy.uint32]
		shapes = [(10,), (), (4, 3, 2), (1,), (2, 2)]
		MemoryMap.create('Test/test_offsets.memmap', dtypes, shapes)
		offsets = [x.offset for x in MemoryMap.open('Test/test_offsets.memmap')]
		# 18 8-byte numbers/strings in the header = 144 for first offset
		# 10 1-byte numbers in first array = +10
		# 0 2-byte numbers in second array = +0
		# 24 4-byte numbers in third array = +96
		# 1 50-byte string in fourth array = +50
		# 4 4-byte numbers in fifth array finishes out the file
		self.assertEqual(offsets, [144, 154, 154, 250, 300])

if __name__ == '__main__':
	unittest.main()
