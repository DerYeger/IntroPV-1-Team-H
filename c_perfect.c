#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define UNCHECKED 0
#define NON_PRIME 1
#define PRIME 2

#define RUNTIME 30
#define RANGE 10

#define NUM_THREADS 32

#define LIMIT 100000000L

unsigned long *primes;
unsigned long ready_limit;
unsigned long last_printed;
unsigned long prime_count;

unsigned long provider = 2;

bool *done;
bool terminate;

//wait until all previous numbers are done or terminate flag is set
void wait_for_print(const unsigned long number)
{
	bool my_terminate = false;
	unsigned long my_last_printed = 1;

	for (unsigned long i = number - 1; i > my_last_printed; i--)
	{
		bool is_done = false;
		
		while (!is_done && !my_terminate)
		{
			#pragma omp atomic read
			is_done = done[i];

			#pragma omp atomic read
			my_terminate = terminate;
		}

		#pragma omp atomic read
		my_last_printed = last_printed;
	}
}

bool is_prime(const unsigned long number)
{
	unsigned long lower_bound = 0;

	unsigned long upper_bound;
	#pragma omp atomic read
	upper_bound = prime_count;

	upper_bound--;

	while (lower_bound <= upper_bound)
	{
		const unsigned long middle = lower_bound + (upper_bound - lower_bound) / 2;

		unsigned long value;
		#pragma omp atomic read
		value = primes[middle];

		if (number < value)
		{
			upper_bound = middle - 1;
		}
		else if (value < number)
		{
			lower_bound = middle + 1;
		}	
		else
		{
			return true;
		}		
	}
	return false;
}

bool is_almost_perfect(const unsigned long number, unsigned int *known_primes)
{
	if (is_prime(number))
	{
		known_primes[number] = 2;
		return number - 1 < RANGE;
	}
	else
	{
		known_primes[number] = 1;
	}

	unsigned long prime_offset = 0;
	unsigned long remainder = number;
	unsigned long sum = 1;

	const unsigned long lower_bound = 2 * number > RANGE ? 2 * number - RANGE : 0;
	const unsigned long upper_bound = 2 * number + RANGE;

	while (remainder > 1)
	{
		if (known_primes[remainder] == UNCHECKED)
		{
			known_primes[remainder] = is_prime(remainder) ? PRIME : NON_PRIME;
		}

		if (known_primes[remainder] == PRIME)
		{
			sum *= remainder + 1;
			break;
		}

		unsigned long prime;
		#pragma omp atomic read
		prime = primes[prime_offset++];

		if (remainder % prime != 0)
		{
			continue;
		}

		unsigned long power = prime;
	
		while (remainder % prime == 0)
		{
			remainder /= prime;
			power *= prime;
		}

		sum *= (power - 1) / (prime - 1);

		if (sum >= upper_bound)
		{
			return false;
		}
	}

	return lower_bound < sum && sum < upper_bound;
}

void calculate_almost_perfects()
{	
	unsigned long my_ready_limit = 0;

	bool my_terminate;
	#pragma omp atomic read
	my_terminate = terminate;

	unsigned int *known_primes = calloc(LIMIT, sizeof(int));
	if (known_primes == NULL)
	{
		exit(EXIT_FAILURE);
	}

	while (!my_terminate)
	{
		unsigned long number;
		#pragma omp atomic capture
		number = provider++;

		if (number >= LIMIT)
		{
			break;
		}

		while (my_ready_limit < number && !my_terminate)
		{
			#pragma omp atomic read
			my_ready_limit = ready_limit;

			#pragma omp atomic read
			my_terminate = terminate;
		}

		if (my_terminate)
		{
			break;
		}

		const bool almost_perfect = is_almost_perfect(number, known_primes);

		if (almost_perfect)
		{
			wait_for_print(number);

			#pragma omp atomic read
			my_terminate = terminate;
			
			if (!my_terminate)
			{
				printf("%lu\n", number);

				#pragma omp atomic write
				last_printed = number;
			}	
		}
					
		#pragma omp atomic write
		done[number] = true;

		#pragma omp atomic read
		my_terminate = terminate;
	}

	free(known_primes);
}

void generate_primes()
{
	unsigned long number = 3;

	bool my_terminate;
	#pragma omp atomic read
	my_terminate = terminate;

	#pragma omp atomic write
	primes[0] = 2;

	#pragma omp atomic update
	prime_count++;

	while (!my_terminate && number < LIMIT)
	{
		bool is_prime = true;

		for (unsigned long i = 1; i < prime_count; i++)
		{
			const unsigned long prime = primes[i];

			if (number % prime == 0)
			{
				is_prime = false;
				break;
			}

			if (prime * prime > number)
			{
				//number is prime
				break;
			}
		}

		if (is_prime)
		{
			#pragma omp atomic write
			primes[prime_count] = number;

			#pragma omp atomic update
			prime_count++;
		}

		number += 2;	

		#pragma omp atomic read
		my_terminate = terminate;

		#pragma omp atomic write
		ready_limit = number - 2;
	}
}

void check_time(const double start_time)
{
	while (omp_get_wtime() - start_time < RUNTIME)
	{
		//do nothing
	}

	#pragma omp atomic write
	terminate = true;
}

int main()
{	
	omp_set_num_threads(NUM_THREADS);

	const double start_time = omp_get_wtime();

	primes = malloc(sizeof(unsigned long) * LIMIT);
	if (primes == NULL)
	{
		return EXIT_FAILURE;
	}

	ready_limit = 0;
	last_printed = 1;
	prime_count = 0;

	done = calloc(LIMIT, sizeof(bool));
	if (done == NULL)
	{
		return EXIT_FAILURE;
	}

	terminate = false;

	#pragma omp parallel default(none) shared(primes, ready_limit, last_printed, provider, done, terminate)
	{
		if (omp_get_thread_num() == 0)
		{
			check_time(start_time);
		}
		else if (omp_get_thread_num() == 1)
		{
			generate_primes();
		}
		else
		{
			calculate_almost_perfects();
		}
	}

	#pragma omp parallel
	{
		printf("\nBye from thread %d", omp_get_thread_num());
	}

	//for debugging
	// printf("\n\nReady up to: %lu\n", ready_limit);
	// printf("Tested up to %lu\n", provider);

	return EXIT_SUCCESS;
}