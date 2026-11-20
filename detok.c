/*
 * NNCP preprocessor, decode path only ("preprocess d" in preprocess.c).
 *
 * Copyright (c) 2018-2019 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/* usage: detok dictfile infile outfile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *word[65536];
static int word_len[65536];

int main(int argc, char **argv)
{
    unsigned char buf[4096];
    int n = 0, len = 0, c, hi, lo, i, type = 0, type1 = 0, space = 0;
    FILE *dict, *in, *out;

    if (argc != 4)
        return fputs("usage: detok dictfile infile outfile\n", stderr), 1;
    if (!(dict = fopen(argv[1], "rb")) || !(in = fopen(argv[2], "rb")) ||
        !(out = fopen(argv[3], "wb")))
        return perror("detok"), 1;
    while ((c = getc(dict)) >= 0) {
        if (c == '\n') {
            if (len > 0 && n < 65536) {
                word[n] = memcpy(malloc(len), buf, len);
                word_len[n++] = len;
            }
            len = 0;
            continue;
        }
        if (c == '\\') {
            if ((c = getc(dict)) < 0)
                break;
            if (c == 'n')
                c = '\n';
            else if (c != '\\')
                return fputs("Invalid escape\n", stderr), 1;
        }
        if (len >= (int)sizeof(buf))
            return fputs("Word too long\n", stderr), 1;
        buf[len++] = c;
    }
    while ((hi = getc(in)) >= 0 && (lo = getc(in)) >= 0) {
        c = hi << 8 | lo;
        if (c >= n)
            return fprintf(stderr, "Invalid symbol %d\n", c), 1;
        for (i = 0; i < word_len[c]; i++) {
            int ch = word[c][i];
            if (ch == 2 || ch == 3) {          /* to upper / first upper */
                type = ch;
            } else if (ch == 1) {              /* no space */
                space = 0;
            } else if (ch == ' ') {
                type1 = type;
                type = 0;
                if (space)
                    putc(' ', out);
                space = 1;
            } else {
                if (type1 == 2 || type1 == 3) {
                    if (ch >= 'a' && ch <= 'z')
                        ch += 'A' - 'a';
                    if (type1 == 3)
                        type1 = 0;
                }
                putc(ch, out);
            }
        }
    }
    return fclose(out) != 0;
}
