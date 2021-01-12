#include <iostream>
#include <omp.h>

int main(int argc, char **argv) {
    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "Hello World from thread: ";
            std::cout << omp_get_thread_num();
            std::cout << "!" << std::endl;
        }
    }
    
    return 0;
}
