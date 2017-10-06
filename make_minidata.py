# coding: utf-8

import codecs

SOURCE_PATH = "data/source.txt"
TARGET_PATH = "data/target.txt"

MINI_SOURCE_PATH = "data/source_mini.txt"
MINI_TARGET_PATH = "data/target_mini.txt"

def make_minidata():
    
    slines = codecs.open(SOURCE_PATH, "r", "utf-8").read().split("\n")[:1000]
    tlines = codecs.open(TARGET_PATH, "r", "utf-8").read().split("\n")[:1000]

    s = codecs.open(MINI_SOURCE_PATH, "w", "utf-8")
    t = codecs.open(MINI_TARGET_PATH, "w", "utf-8")

    for i in range(len(slines)):
        s.write(slines[i]+"\n")
        t.write(tlines[i]+"\n")

    s.close()
    t.close()

if __name__ == "__main__":
    make_minidata()