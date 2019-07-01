#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void print_array(int **a, const int length)
{
	for (int y = 0; y < length; y++)
	{
		for (int x = 0; x < length; x++)
		{
			printf("%d ", a[y][x]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(const int argc, const char *argv[])
{
	if (argc < 3)
	{
		printf("Additional arguments required\n");
		return -1;
	}

	const int n = atoi(argv[1]);
	const int R = atoi(argv[2]);

	if (argc >= 4)
	{
		srand(atoi(argv[3]));
	}
	else
	{
		srand(123);
	}

	const double start_time = omp_get_wtime();

	const int n_threads = omp_get_max_threads();

	int **a = malloc(sizeof(int *) * n);

	int *round_info = malloc(sizeof(int) * R);
	int ***round_impacts = malloc(sizeof(int **) * R);

	for (int r = 0; r < R; r++)
	{
		const int k = rand() % 10;
	
		round_info[r] = k;
		round_impacts[r] = malloc(sizeof(int *) * k);

		for (int w = 0; w < k; ++w)
		{
			const int z = rand() % (n/4); // int-Division
			const int i = rand() % n; //y
			const int j = rand() % n; //x

			round_impacts[r][w] = malloc(sizeof(int) * 3);

			round_impacts[r][w][0] = z;
			round_impacts[r][w][1] = i;
			round_impacts[r][w][2] = j;
		}
	}	
	
	int max = 0;
	
	#pragma omp parallel default(none) shared(a) firstprivate(round_info, round_impacts) reduction(max:max)
	{
		#pragma omp for schedule(static, 1) nowait
		for (int y = 0; y < n; y++)
		{
			a[y] = calloc(n, sizeof(int));
		}	

		for (int r = 0; r < R; ++r)
		{
			if (n <= 16)
			{
				#pragma omp master
				{
					printf("Eingeschlagen:");
				}	
			}

			const int k = round_info[r];
			
			for (int w = 0; w < k; ++w)
			{
				const int z = round_impacts[r][w][0];
				const int i = round_impacts[r][w][1];
				const int j = round_impacts[r][w][2];

				if (n <= 16)
				{
					#pragma omp master
					{
						printf(" %d(%d, %d)", z, i, j);
					}	
				}

				const int reach = z - 1;
				const int lower_x_bound = (j - reach > 0) ? j - reach : 0;
				const int upper_x_bound = (reach + j < n) ? reach + j : n - 1;
				const int lower_y_bound = (i - reach > 0) ? i - reach : 0;
				const int upper_y_bound = (reach + i < n) ? reach + i : n - 1;

				const int alignment_offset = lower_y_bound % n_threads;
				const int alignment = lower_y_bound - alignment_offset;

				#pragma omp for schedule(static, 1) nowait
				for (int y = alignment; y <= upper_y_bound; y++)
				{
					if (y < lower_y_bound)
					{
						continue;
					}

					const int y_distance = abs(y - i);
					const int constant_start = (j - y_distance < lower_x_bound) ? lower_x_bound : j - y_distance;
					const int constant_end = (j + y_distance > upper_x_bound) ? upper_x_bound : j + y_distance;
					const int constant_value = z - y_distance;

					for (int x = lower_x_bound; x < constant_start; x++)
					{
						a[y][x] += z - j + x;
					}

					for (int x = constant_start; x <= constant_end; x++)
					{
						a[y][x] += constant_value;
					}

					for (int x = constant_end + 1; x <= upper_x_bound; x++)
					{
						a[y][x] += z + j - x;
					}
				}
			}

			if (n <= 16) 
			{
				#pragma omp barrier
				#pragma omp master
				{
					printf("\n");
					print_array(a, n);
				}
			}
			#pragma omp barrier
		}

		#pragma omp for schedule(static, 1)
		for (int y = 0; y < n; y++)
		{
			for (int x = 0; x < n; x++)
			{
				if (a[y][x] > max)
				{
					max = a[y][x];
				}
			}
		}
	}

	const int end_time = omp_get_wtime();

	printf("time = %fs\n", end_time - start_time);
	printf("max = %d\n", max);
	if (n > 80)
	{
		printf("a[80][15] = %d\n", a[80][15]);
	}

	// int i, j;
	// printf("Gewuenschte i-Koordinate = ");
	// scanf("%d", &i);
	// printf("Gewuenschte j-Koordinate = ");
	// scanf("%d", &j);
	// if (0 < i && i < n && 0 < j && j < n)
	// {
	// 	printf("a[%d][%d]=%d\n", i, j, a[i][j]);
	// }
	// else
	// {
	// 	printf("Ungueltige Koordianten\n");
	// }
	
	return 0;
}