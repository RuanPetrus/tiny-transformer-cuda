#include "nn.cu"

#define DATASET_PATH "./dataset/tinyshakespeare.txt"
#define return_defer(res) { result = res; goto defer; }

char *read_entire_file(const char *path, long long *sz)
{
	bool result = true;
	long long m;
	char *content;

    FILE *f = fopen(path, "rb");
    if (f == NULL)  return_defer(false);
    if (fseek(f, 0, SEEK_END) < 0) return_defer(false);

#ifndef _WIN32
    m = ftell(f);
#else
    m = _ftelli64(f);
#endif
    if (m < 0)                     return_defer(false);
    if (fseek(f, 0, SEEK_SET) < 0) return_defer(false);

	content = (char *) malloc(m * sizeof(*content));
    fread(content, m, 1, f);
    if (ferror(f)) {
        return_defer(false);
    }

defer:
    if (!result) ERROR("Could not read file %s: %s", path, strerror(errno));
    if (f) fclose(f);
	*sz = m;
    return content;
}

int main()
{
	long long dataset_cnt;
	char* dataset_text = read_entire_file(DATASET_PATH, &dataset_cnt);

	long long char_cnt[256] = {0};
	for (int i = 0; i < dataset_cnt; i++) {
		char_cnt[dataset_text[i]] += 1;
	}
	char alphabet[256] = {0};
	int alphabet_n = 0;
	int stoi[256] = {0};
	for (int c = 0; c < 256; c++) {
		if (char_cnt[c]) {
			int id = alphabet_n++;
			alphabet[id] = c;
			stoi[c] = id;
		}
	}
	printf("dataset_text (%lld)\n", dataset_cnt);
	printf("-----------------\n");
	for (int i = 0; i < min(dataset_cnt, 100ll); i++) {
		char c = dataset_text[i];
		printf("%c", c);
	}
	printf("\n");
	printf("-----------------\n");

	char *itos = alphabet;
	printf("Alphabet (%d):\n", alphabet_n);
	printf("-----------------\n");
	for (int i = 0; i < alphabet_n; i++) {
		char c = alphabet[i];
		if (c == '\n') {
			printf("\\n");
		}
		else {
			printf("%c", c);
		}
	}
	printf("\n");
	printf("-----------------\n");

	Config config = {
		1, alphabet_n, 3, 1
	};

	const int mx_token = 100;
	Transformer mm(config);
	int *tokens = mm.generate(mx_token);

	for (int i = 0; i < mx_token; i++) {
		printf("%c", itos[tokens[i]]);
	}
	printf("\n");

	return 0;
}
