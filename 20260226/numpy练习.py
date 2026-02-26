import numpy

a = numpy.array([10, 20, 30])
print(a)

b = numpy.array([1, 2, 3])
print(a + b)

print(b * 5)

c = numpy.array([2, 4, 6, 8])
print(numpy.mean(c))

d = numpy.array([4,5,6])
print(numpy.dot(b,d))

e = numpy.array([[1,2],[3,4]])
print(e)
print(e.shape)

f = numpy.array([5,6])

print(numpy.dot(e,f))


w = numpy.array([0.5,-1])
x = numpy.array([2,3])
y = numpy.dot(w,x)
print(y)