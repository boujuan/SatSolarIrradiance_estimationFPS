# import struct
import codecs

path = r'data\RR1720_128312\data\raw\os150\rr2017_290_47008.raw'

with open(path, 'rb') as file:
    binary_data = file.read()
    
u_string = codecs.decode(binary_data, 'utf-8')
print(u_string)
