"""Send a single test signal to the Arduino to check the serial connection."""
import time

import serial

SERIAL_PORT = "/dev/tty.usbmodem101"  # Arduino port (check with `ls /dev/tty.*`)
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Wait for the connection to establish

ser.write(b"1")
print("Sent signal to Arduino!")

ser.close()
