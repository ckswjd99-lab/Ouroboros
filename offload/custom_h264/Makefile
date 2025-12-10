CC      := gcc
PKGS    := libavcodec libavutil libswscale
CFLAGS  := -fPIC -O2 -Wall -Wextra $(shell pkg-config --cflags $(PKGS))
LDFLAGS := -shared $(shell pkg-config --libs $(PKGS))

all: libcustom_h264.so

libcustom_h264.so: custom_h264.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f libcustom_h264.so
