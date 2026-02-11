#include<stdlib.h>

int f(int *list, int val, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        list[i] += val;
    }
    return list[n-1];
}

int main() {
    int data[5] = {1, 2, 3, 4, 5};
    return f(data, 10, 5);
}