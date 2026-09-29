/* Convert a binary file to a C byte array for embedding in msieve. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--text] symbol input\n", prog);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    const char *symbol;
    const char *filename;
    FILE *in;
    unsigned char buf[4096];
    size_t n, i;
    unsigned long long count = 0;
    int text = 0;
    unsigned int col = 0;

    if (argc == 4 && strcmp(argv[1], "--text") == 0) {
        text = 1;
        symbol = argv[2];
        filename = argv[3];
    }
    else if (argc == 3) {
        symbol = argv[1];
        filename = argv[2];
    }
    else {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    in = fopen(filename, "rb");
    if (in == NULL) {
        perror(filename);
        return EXIT_FAILURE;
    }

    printf("/* Generated from %s; do not edit. */\n", filename);
    printf("#if defined(_MSC_VER)\n");
    printf("#define MSIEVE_EMBED_ALIGN __declspec(align(16))\n");
    printf("#elif defined(__GNUC__)\n");
    printf("#define MSIEVE_EMBED_ALIGN __attribute__((aligned(16)))\n");
    printf("#else\n");
    printf("#define MSIEVE_EMBED_ALIGN\n");
    printf("#endif\n\n");
    printf("MSIEVE_EMBED_ALIGN const unsigned char %s[] = {\n", symbol);

    while ((n = fread(buf, 1, sizeof(buf), in)) != 0) {
        for (i = 0; i < n; i++) {
            if (col == 0)
                printf("    ");
            printf("0x%02x,", (unsigned int)buf[i]);
            count++;
            col++;
            if (col == 12) {
                printf("\n");
                col = 0;
            }
            else {
                printf(" ");
            }
        }
    }

    if (ferror(in)) {
        perror(filename);
        fclose(in);
        return EXIT_FAILURE;
    }
    fclose(in);

    /* cuModuleLoadData() expects PTX input to be NUL-terminated. */
    if (text) {
        if (col == 0)
            printf("    ");
        printf("0x00,");
        count++;
        col++;
    }
    if (col != 0)
        printf("\n");

    printf("};\n");
    printf("const unsigned long long %s_size = %lluULL;\n", symbol, count);
    printf("#undef MSIEVE_EMBED_ALIGN\n");
    return EXIT_SUCCESS;
}
