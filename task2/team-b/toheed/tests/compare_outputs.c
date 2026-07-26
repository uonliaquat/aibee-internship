#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 8192
#define MAX_IDS  4096

/* Parse one line of space-separated ints into ids[]. Returns count, or -1 on error. */
static int parse_ids(const char *line, int *ids, int max_ids) {
    int count = 0;
    const char *p = line;

    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') break;

        char *end = NULL;
        long val = strtol(p, &end, 10);
        if (end == p) {
            /* non-number junk on the line */
            return -1;
        }

        if (count >= max_ids) return -1;
        ids[count++] = (int)val;
        p = end;
    }
    return count;
}

static int lines_equal(const int *a, int na, const int *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void print_ids(const char *label, const int *ids, int n) {
    printf("  %s (%d):", label, n);
    for (int i = 0; i < n; i++) {
        printf(" %d", ids[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const char *hf_path = "tests/hf_tokens_normalized.txt";
    const char *c_path  = "output.txt";

    if (argc >= 3) {
        hf_path = argv[1];
        c_path  = argv[2];
    } else if (argc == 2) {
        printf("Usage: %s [hf_file c_file]\n", argv[0]);
        printf("Default: %s  vs  %s\n", hf_path, c_path);
        return 1;
    }

    FILE *hf = fopen(hf_path, "r");
    FILE *cf = fopen(c_path, "r");
    if (!hf) {
        fprintf(stderr, "Could not open HF file: %s\n", hf_path);
        return 1;
    }
    if (!cf) {
        fprintf(stderr, "Could not open C output file: %s\n", c_path);
        fclose(hf);
        return 1;
    }

    char hf_line[MAX_LINE];
    char c_line[MAX_LINE];
    int hf_ids[MAX_IDS];
    int c_ids[MAX_IDS];

    int line_no = 0;
    int passed = 0;
    int failed = 0;

    while (1) {
        char *hf_ok = fgets(hf_line, sizeof(hf_line), hf);
        char *c_ok  = fgets(c_line, sizeof(c_line), cf);

        if (!hf_ok && !c_ok) break; /* both EOF — done */

        line_no++;

        if (!hf_ok || !c_ok) {
            failed++;
            printf("FAIL line %d: one file ended early\n", line_no);
            if (!hf_ok) printf("  HF file ended, C file still has lines\n");
            if (!c_ok)  printf("  C file ended, HF file still has lines\n");
            break;
        }

        /* strip trailing newline */
        hf_line[strcspn(hf_line, "\r\n")] = '\0';
        c_line[strcspn(c_line, "\r\n")] = '\0';

        int nh = parse_ids(hf_line, hf_ids, MAX_IDS);
        int nc = parse_ids(c_line, c_ids, MAX_IDS);

        if (nh < 0 || nc < 0) {
            failed++;
            printf("FAIL line %d: could not parse ids\n", line_no);
            printf("  HF: %s\n", hf_line);
            printf("  C : %s\n", c_line);
            continue;
        }

        if (lines_equal(hf_ids, nh, c_ids, nc)) {
            passed++;
        } else {
            failed++;
            printf("FAIL line %d (sentence %d)\n", line_no, line_no);
            print_ids("HF", hf_ids, nh);
            print_ids("C ", c_ids, nc);
        }
    }

    fclose(hf);
    fclose(cf);

    printf("\n==============================\n");
    printf("Compared %d sentences\n", line_no);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);
    printf("==============================\n");

    return failed == 0 ? 0 : 1;
}
