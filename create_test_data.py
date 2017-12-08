import rstr
from random import SystemRandom


bookingtype_list = ['Barentnahme']

receiver_regex = rstr.xeger(r'GA NR0000[0-9]{4}BLZ[0-9]{8}')
print(receiver_regex)

purpose_regex1 = rstr.xeger(r'[0-9]{2}\.[0-9]{2}/[0-9]{2}:[0-9]{2}UHR ([A-Z]{4}|[A-Z]{8}|[A-Z]{10})')
purpose_regex2 = rstr.xeger(r'PGA [0-9]{8} KRT[0-9]{4}/[0-9]{2}\.[0-9]{2} [0-9]{2}\.[0-9]{2} [0-9]{2}\.[0-9]{2} TA-NR\. [0-9]{6} ([A-Z]{4}|[A-Z]{8}|[A-Z]{10})')

print(purpose_regex1)
print(purpose_regex2)

