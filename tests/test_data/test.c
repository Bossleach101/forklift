#include <stdio.h>

extern void i64(int *list, int val, int n);

int main() {
    int list[] = {1, 2, 3, 4, 5};
    int n = 5;
    int val = 10;

    i64(list, val, n);

    for (int i = 0; i < n; ++i) {
        printf("%d ", list[i]);
    }

    return 0;
}
