#include <iostream>
#include<cstdio>
#include<cmath>
#include "mpi.h"
#include<omp.h>
#include <chrono>


const double  PI = 3.1415926535897932;
double lambda;
double L = 1.;
double eps = 0.000001;
double NormMatr1(double* A, double* B, int n, int m)
{
    double s, max;
    max = 0;
    for (int i = 0; i < n; i++)
    {

        for (int j = 0; j < m; j++)
        {
            s = std::fabs(A[i * m + j] - B[i * m + j]);
            if (max < s)
            {
                max = s;
            }
        }
    }
    return max;
}
double F(double x, double y)
{
    return (1 - x) * x * std::sin(PI * y);
    return std::sinh(x) + std::sinh(y);
}
double Granpr(double y)
{
    return 0;
    return std::sinh(y) + std::sinh(L);
}
double Granlf(double y)
{
    return 0;
    return std::sinh(y) + std::sinh(0);
}
double Grannz(double x)
{
    return 0;
    return std::sinh(x) + std::sinh(0);
}
double Granvh(double x)
{
    return 0;
    return std::sinh(x) + std::sinh(L);
}
double PR(double x, double y)
{
    return 2 * std::sin(PI * y) + lambda * (1 - x) * x * std::sin(PI * y) + PI * PI * (1 - x) * x * std::sin(PI * y);
}
void mpitest(int argc, char** argv)
{

    int myid, np;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    // Общее число процессов в рамках задачи
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Номер текущего процесса в рамках задачи
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);
    //printf("Process %d of %d is on %s\n", myid, np, processor_name);
    fflush(stdout);
    int x = 0, y = 0;
    //printf("before: id = %d, x = %d, y = %d\n", myid, x,y);
    if (myid == 0)
    {
        MPI_Send(&x, 1, MPI_INT, 1, 42, MPI_COMM_WORLD);
        MPI_Send(&y, 1, MPI_INT, 1, 43, MPI_COMM_WORLD);

    }
    if (myid == 1)
    {
        MPI_Recv(&y, 1, MPI_INT, 0, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&x, 1, MPI_INT, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    }
    fflush(stdout);
    // printf("after: id = %d, x = %d, y = %d\n", myid, x,y);
     //std::cout << "TA-DA\n";
    int a = -100;
    int b = myid;
    int c = 100;
    int dst = (myid == 0) ? np - 1 : myid - 1;
    int src = (myid == np - 1) ? 0 : myid + 1;
    int sendCount = (myid == 0) ? 0 : 1;
    int recvCount = (myid == np - 1) ? 0 : 1;
    MPI_Sendrecv(&b, sendCount, MPI_INT, dst, 0, &c, recvCount, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < np; i++)
    {
        if (myid == i)
        {
            printf("myid = % d \t a=%d \t b=%d \t c=%d\n", myid, a, b, c);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myid == 0)
        printf("____________________________________\n");
    MPI_Sendrecv(&b, recvCount, MPI_INT, src, 0, &a, sendCount, MPI_INT, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < np; i++)
    {
        if (myid == i)
        {
            printf("myid = % d \t a=%d \t b=%d \t c=%d\n", myid, a, b, c);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myid == 0)
        printf("____________________________________\n");
    b = 0;
    if (myid == 0)
        b = 100;

    // MPI_Sendrecv(&b, recvCount, MPI_INT, src, 0, &a, sendCount, MPI_INT, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < np; i++)
    {
        if (myid == i)
        {
            printf("myid = %d\t b=%d \n", myid, b);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myid == 0)
        printf("__________________________ \n");
    MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < np; i++)
    {
        if (myid == i)
        {
            printf("myid = %d\t b=%d \n", myid, b);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int A[1000];
    int B[1000];
    if (myid == 0)
    {
        for (int i = 0; i < 1000; i++)
        {
            A[i] = i + 1;
            B[i] = -(i + 1);
        }
    }
    if (myid == 1)
    {
        for (int i = 0; i < 1000; i++)
        {
            A[i] = 0;
            B[i] = 0;
        }
    }
    MPI_Request send, recv, send1, recv1;
    if (myid == 0)
    {
        MPI_Send_init(A, 1000, MPI_INT, 1, 117, MPI_COMM_WORLD, &send);
        MPI_Send_init(B, 1000, MPI_INT, 1, 110, MPI_COMM_WORLD, &send1);
    }
    if (myid == 1)
    {
        MPI_Recv_init(A, 1000, MPI_INT, 0, 117, MPI_COMM_WORLD, &recv);
        MPI_Recv_init(B, 1000, MPI_INT, 0, 110, MPI_COMM_WORLD, &recv1);
    }
    if (myid == 0)
    {
        MPI_Start(&send);
        MPI_Start(&send1);
    }
    if (myid == 1)
    {
        MPI_Start(&recv);
        MPI_Start(&recv1);
    }
    if (myid == 1)
    {
        std::cout << A[10] << ' ' << B[0] << std::endl;
    }
    if (myid == 1)
        std::cout << "____________________________________" << std::endl;
    if (myid == 1)
    {
        MPI_Wait(&recv, MPI_STATUS_IGNORE);
    }
    if (myid == 1)
        std::cout << "____________________________________" << std::endl;
    if (myid == 1)
    {
        std::cout << A[10] << ' ' << B[10] << std::endl;
    }
    if (myid == 1)
        std::cout << "____________________________________" << std::endl;
    if (myid == 1)
    {
        MPI_Wait(&recv1, MPI_STATUS_IGNORE);
    }
    if (myid == 1)
        std::cout << "____________________________________" << std::endl;
    if (myid == 1)
    {
        std::cout << A[10] << ' ' << B[10] << std::endl;
    }


}
void testZeidel(int n, int myid)
{
    auto begin = std::chrono::steady_clock::now();
        double h = L / (n - 1);
        lambda = (1 / h) * (1 / h);
        double* u;
        double* ures;
        u = new double[n * n];
        double* A;
        A = new double[n * n];
        ures = new double[n * n];
           double t;
        double res = 1;
        int iter = 0;
        iter = 0;
        for (int i = 0; i < n * n; i++)
        {
            u[i] = 1.00001;
            ures[i] = 1.00001;
        }
        for (int i = 0; i < n; i++)
        {
            u[n * (n - 1) + i] = Granvh(i * h);
            u[i * n] = Granlf(i * h);
            u[i] = Grannz(i * h);
            u[i * n + n - 1] = Granpr(i * h);
            ures[n * (n - 1) + i] = Granvh(i * h);
            ures[i * n] = Granlf(i * h);
            ures[i] = Grannz(i * h);
            ures[i * n + n - 1] = Granpr(i * h);
        }
        iter = 0;
       
        int ch;
        while (res > eps)
        {


            for (int j = 1; j < n - 1; j++)
                for (int i = (j%2==1) ? 2 : 1; i < n - 1; i+=2)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
            for (int j = 1; j < n - 1; j++)
                for (int i = (j % 2 == 1) ? 1 : 2; i < n - 1; i+=2)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
            iter++;
            res = 0;
            for (int i = 0; i < n * n; i++)
                res += (u[i] - ures[i]) * (u[i] - ures[i]);
            res = std::sqrt(res);
            std::swap(u, ures);
        }
        
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << ures[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "ZEIDEL NO_MPI " << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(u, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
        std::cout << "Time " << elapsed_ms.count() << std::endl;
        delete[] u;
        delete[] A;
        delete[] ures;
   

}
void testZeidel_if(int n, int myid)
{
    auto begin = std::chrono::steady_clock::now();
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    u = new double[n * n];
    double* A;
    A = new double[n * n];
    ures = new double[n * n];
    double t;
    double res = 1;
    int iter = 0;
    iter = 0;
    for (int i = 0; i < n * n; i++)
    {
        u[i] = 1.00001;
        ures[i] = 1.00001;
    }
    for (int i = 0; i < n; i++)
    {
        u[n * (n - 1) + i] = Granvh(i * h);
        u[i * n] = Granlf(i * h);
        u[i] = Grannz(i * h);
        u[i * n + n - 1] = Granpr(i * h);
        ures[n * (n - 1) + i] = Granvh(i * h);
        ures[i * n] = Granlf(i * h);
        ures[i] = Grannz(i * h);
        ures[i * n + n - 1] = Granpr(i * h);
    }
    iter = 0;

    int ch;
    while (res > eps)
    {


        for (int j = 1; j < n - 1; j++)
            for (int i = 1; i < n - 1; i ++)
                if((i+j)%2==1)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        for (int j = 1; j < n - 1; j++)
            for (int i = 1; i < n - 1; i ++)
                if ((i + j) % 2 == 0)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
        for (int i = 0; i < n * n; i++)
            res += (u[i] - ures[i]) * (u[i] - ures[i]);
        res = std::sqrt(res);
        std::swap(u, ures);
    }

    if (n < 20)
    {
        std::cout << "______________________" << std::endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << ures[i * n + j] << ' ';
            }
            std::cout << std::endl;
        }

        std::cout << "______________________" << std::endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << F(i * h, j * h) << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "______________________" << std::endl;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = F(i * h, j * h);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "ZEIDEL NO_MPI " << " matrix size " << n - 2 << std::endl;
    std::cout << "Norm C " << NormMatr1(u, A, n, n) << std::endl;
    std::cout << "Count iter " << iter << std::endl;
    std::cout << "Time " << elapsed_ms.count() << std::endl;
    delete[] u;
    delete[] A;
    delete[] ures;


}
void testJacobi(int n, int myid)
{
    auto begin = std::chrono::steady_clock::now();
        double h = L / (n - 1);
        lambda = (1 / h) * (1 / h);
        double* u;
        double* ures;
        u = new double[n * n];
        double* A;
        A = new double[n * n];
        ures = new double[n * n];
        for (int i = 0; i < n * n; i++)
        {
            u[i] = 1.00001;
            ures[i] = 1.00001;
        }
        for (int i = 0; i < n; i++)
        {
            u[n * (n - 1) + i] = Granvh(i * h);
            u[i * n] = Granlf(i * h);
            u[i] = Grannz(i * h);
            u[i * n + n - 1] = Granpr(i * h);
            ures[n * (n - 1) + i] = Granvh(i * h);
            ures[i * n] = Granlf(i * h);
            ures[i] = Grannz(i * h);
            ures[i * n + n - 1] = Granpr(i * h);
        }
        double t;
        double res = 1;
        int iter = 0;
        iter = 0;
        
        
        while (res > eps)
        {


            for (int j = 1; j < n - 1; j++)
                for (int i = 1; i < n - 1; i++)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
            iter++;
            res = 0;
            /*for (int j = 1; j < n - 1; j++)
                for (int i = 1; i < n - 1; i++)
                    res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);*/
            for (int i = 0; i < n * n; i++)
                res += (u[i] - ures[i]) * (u[i] - ures[i]);
            res = std::sqrt(res);
            std::swap(u, ures);
        }
        
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << ures[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "JACOBI NO_MPI " << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(u, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
        std::cout << "Time " << elapsed_ms.count() << std::endl;
   
        delete[] u;
        delete[] A;
        delete[] ures;
}
void testompJacobi_Zeidel(int n, int myid)
{
    if (myid == 0)
    {
        double t1, t2;
        
        double h = L / (n - 1);
        double x, y;
        lambda = (1 / h) * (1 / h);
        double* u;
        double* ures;
        u = new double[n * n];
        double* A;
        A = new double[n * n];
        ures = new double[n * n];
        double res = 1;
        int iter;
        for (int i = 0; i < n * n; i++)
        {
            u[i] = 1.00001;
            ures[i] = 1.00001;
        }
        for (int i = 0; i < n; i++)
        {
            u[n * (n - 1) + i] = Granvh(i * h);
            u[i * n] = Granlf(i * h);
            u[i] = Grannz(i * h);
            u[i * n + n - 1] = Granpr(i * h);
            ures[n * (n - 1) + i] = Granvh(i * h);
            ures[i * n] = Granlf(i * h);
            ures[i] = Grannz(i * h);
            ures[i * n + n - 1] = Granpr(i * h);
        }
        iter = 0;
        t1 = MPI_Wtime();
        while (res > eps)
        {
            iter++;
            std::swap(u, ures);
#pragma omp parallel for
            for (int j = 1; j < n - 1; j++)
                for (int i = 1; i < n - 1; i++)
                    ures[i + j * n] = (PR(i * h, j * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
            res = 0;
            for (int j = 1; j < n -1; j++)
                for (int i = 1; i < n - 1; i++)
                    res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
            //for (int i = 0; i < n * n; i++)
              //  res += (u[i] - ures[i]) * (u[i] - ures[i]);
            res = std::sqrt(res);
        }
        t2 = MPI_Wtime();
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << ures[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        
        std::cout << "JACOBI NO_MPI OMP " << " Threads of nodes " << omp_get_num_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(u, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
        std::cout << "Time " << t2 - t1 << std::endl;
        
        res = 1;
        for (int i = 0; i < n * n; i++)
        {
            u[i] = 1.00001;
            ures[i] = 1.00001;
        }
        for (int i = 0; i < n; i++)
        {
            u[n * (n - 1) + i] = Granvh(i * h);
            u[i * n] = Granlf(i * h);
            u[i] = Grannz(i * h);
            u[i * n + n - 1] = Granpr(i * h);
            ures[n * (n - 1) + i] = Granvh(i * h);
            ures[i * n] = Granlf(i * h);
            ures[i] = Grannz(i * h);
            ures[i * n + n - 1] = Granpr(i * h);
        }
        iter = 0;
        t1 = MPI_Wtime();
        while (res > eps)
        {
            iter++;
            std::swap(u, ures);
#pragma omp parallel for
            for (int j = 1; j < n - 1; j++)
                for (int i = 1; i < n - 1; i++)
                    if ((i + j) % 2 == 1)
                        ures[i + j * n] = (PR(i * h, j * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
#pragma omp parallel for
            for (int j = 1; j < n - 1; j++)
                for (int i = 1; i < n - 1; i++)
                    if ((i + j) % 2 == 0)
                        ures[i + j * n] = (PR(i * h, j * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
            res = 0;
            for (int i = 0; i < n * n; i++)
                res += (u[i] - ures[i]) * (u[i] - ures[i]);
            res = std::sqrt(res);
        }
        t2 = MPI_Wtime();
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << ures[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        
        std::cout << "ZEIDEL OMP NO_MPI " << " Threads of nodes " << omp_get_num_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(u, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
        std::cout << "Time " << t2 - t1 << std::endl;
        delete[] u;
        delete[] A;
        delete[] ures;
    }

}
void testJacobiMPISend_Recv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    while (sflag >eps)
    {
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 0; i < n; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI SEND_RECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testJacobiMPISR(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda =(1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    int dst = (myid == 0) ? np - 1 : myid - 1;
    int src = (myid == np - 1) ? 0 : myid + 1;
    int sendCount = (myid == 0) ? 0 : n;
    int recvCount = (myid == np - 1) ? 0 : n;
    
    while (sflag > eps)
    {
        MPI_Sendrecv(&(u[n]), sendCount, MPI_DOUBLE, dst, 117, &(u[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(u[m * n]), recvCount, MPI_DOUBLE, src, 0, &(u[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);

    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI SENDRECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testJacobiMPIISend_IRecv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    MPI_Request senddown0, recvdown0, sendup0, recvup0, senddown1, recvdown1, sendup1, recvup1;
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD, &senddown0);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvdown0);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD, &sendup0);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvup0);
        }
    }
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 116, MPI_COMM_WORLD, &senddown1);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(ures[0]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvdown1);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[n]), n, MPI_DOUBLE, i - 1, 116, MPI_COMM_WORLD, &sendup1);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvup1);
        }
    }
    iter = 0;
    
    while (sflag > eps)
    {
        if (iter % 2 == 0)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
        for (int j = 2; j < m; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 0)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
        for (int i = 1; i < n - 1; i++)
            ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + u[i + 1 + n] + u[i - 1 + n] + u[i + (1 + 1) * n] + u[i + (1 - 1) * n]) / (4 + lambda * h * h);
        for (int i = 1; i < n - 1; i++)
            ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + u[i + 1 + m * n] + u[i - 1 + m * n] + u[i + (m + 1) * n] + u[i + (m - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI ISEND_IRECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPISend_Recv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    
    while (sflag > eps)
    {
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i+=2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(ures[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(ures[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 : 2; i < n - 1; i+=2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL SEND_RECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;

    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPISR(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    int ch = m % 2;
    if (ch != 0)
    {
        ch = (ch + myid) % 2;
    }
    int dst = (myid == 0) ? np - 1 : myid - 1;
    int src = (myid == np - 1) ? 0 : myid + 1;
    int sendCount = (myid == 0) ? 0 : n;
    int recvCount = (myid == np - 1) ? 0 : n;
    
    while (sflag > eps)
    {
        MPI_Sendrecv(&(u[n]), sendCount, MPI_DOUBLE, dst, 117, &(u[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(u[m * n]), recvCount, MPI_DOUBLE, src, 0, &(u[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i += 2)
                    ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        MPI_Sendrecv(&(ures[n]), sendCount, MPI_DOUBLE, dst, 117, &(ures[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(ures[m * n]), recvCount, MPI_DOUBLE, src, 0, &(ures[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 : 2; i < n - 1; i += 2)
                    ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL SENDRECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPIISend_IRecv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda =(1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    MPI_Request senddown0, recvdown0, sendup0, recvup0, senddown1, recvdown1, sendup1, recvup1;
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD, &senddown0);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvdown0);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD, &sendup0);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvup0);
        }
    }
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 116, MPI_COMM_WORLD, &senddown1);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(ures[0]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvdown1);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[n]), n, MPI_DOUBLE, i - 1, 116, MPI_COMM_WORLD, &sendup1);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvup1);
        }
    }
    iter = 0;
    
    while (sflag > eps)
    {
        if (iter % 2 == 0)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
        for (int j = 2; j < m; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i += 2)
                    ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 0)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
        for (int i = ((1 + m * myid) % 2 == 0)? 1:2; i < n - 1; i+=2)
            ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + u[i + 1 + n] + u[i - 1 + n] + u[i + (1 + 1) * n] + u[i + (1 - 1) * n]) / (4 + lambda * h * h);
        for (int i = ((m + m * myid) % 2 == 0) ? 1 : 2; i < n - 1; i += 2)
            ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + u[i + 1 + m * n] + u[i - 1 + m * n] + u[i + (m + 1) * n] + u[i + (m - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 1)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
        for (int j = 2; j < m; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 :2; i < n - 1; i += 2)
                    ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 1)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
        for (int i = ((1 + m * myid) % 2 == 0) ? 2 : 1; i < n - 1; i += 2)
                ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + ures[i + 1 + n] + ures[i - 1 + n] + ures[i + (1 + 1) * n] + ures[i + (1 - 1) * n]) / (4 + lambda * h * h);
        for (int i = ((m + m * myid) % 2 == 0) ? 2 : 1; i < n - 1; i += 2)
                ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + ures[i + 1 + m * n] + ures[i - 1 + m * n] + ures[i + (m + 1) * n] + ures[i + (m - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL ISEND_IRECV " << " NODES " << np << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
        
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
    
}
void testJacobiMPIompSend_Recv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    
    while (sflag > eps)
    {
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 0; i < n; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI SEND_RECV " << " NODES " << np << " Treads " << omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testJacobiMPIompSR(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    int dst = (myid == 0) ? np - 1 : myid - 1;
    int src = (myid == np - 1) ? 0 : myid + 1;
    int sendCount = (myid == 0) ? 0 : n;
    int recvCount = (myid == np - 1) ? 0 : n;
    
    while (sflag > eps)
    {
        MPI_Sendrecv(&(u[n]), sendCount, MPI_DOUBLE, dst, 117, &(u[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(u[m * n]), recvCount, MPI_DOUBLE, src, 0, &(u[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);

    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI SENDRECV " << " NODES " << np << " Treads " << omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testJacobiMPIompISend_IRecv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    MPI_Request senddown0, recvdown0, sendup0, recvup0, senddown1, recvdown1, sendup1, recvup1;
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD, &senddown0);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvdown0);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD, &sendup0);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvup0);
        }
    }
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 116, MPI_COMM_WORLD, &senddown1);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(ures[0]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvdown1);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[n]), n, MPI_DOUBLE, i - 1, 116, MPI_COMM_WORLD, &sendup1);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvup1);
        }
    }
    iter = 0;
    
    while (sflag > eps)
    {
        if (iter % 2 == 0)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
#pragma omp parallel for
        for (int j = 2; j < m; j++)
            for (int i = 1; i < n - 1; i++)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 0)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
        for (int i = 1; i < n - 1; i++)
            ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + u[i + 1 + n] + u[i - 1 + n] + u[i + (1 + 1) * n] + u[i + (1 - 1) * n]) / (4 + lambda * h * h);
        for (int i = 1; i < n - 1; i++)
            ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + u[i + 1 + m * n] + u[i - 1 + m * n] + u[i + (m + 1) * n] + u[i + (m - 1) * n]) / (4 + lambda * h * h);
        iter++;
        t = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                t += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(t);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "JACOBI ISEND_IRECV " << " NODES " << np << " Treads " << omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPIompSend_Recv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    
    while (sflag > eps)
    {
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#pragma omp parallel for
        for(int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        for (int i = 0; i < np - 1; i++)
        {
            if (myid == i)
            {
                MPI_Send(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i + 1)
            {
                MPI_Recv(&(ures[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = np - 1; i > 0; i--)
        {
            if (myid == i)
            {
                MPI_Send(&(ures[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD);
            }
            if (myid == i - 1)
            {
                MPI_Recv(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 : 2; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL SEND_RECV " << " NODES " << np << " Treads " << omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;

    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPIompSR(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    int ch = m % 2;
    if (ch != 0)
    {
        ch = (ch + myid) % 2;
    }
    int dst = (myid == 0) ? np - 1 : myid - 1;
    int src = (myid == np - 1) ? 0 : myid + 1;
    int sendCount = (myid == 0) ? 0 : n;
    int recvCount = (myid == np - 1) ? 0 : n;
    
    while (sflag > eps)
    {
        MPI_Sendrecv(&(u[n]), sendCount, MPI_DOUBLE, dst, 117, &(u[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(u[m * n]), recvCount, MPI_DOUBLE, src, 0, &(u[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        MPI_Sendrecv(&(ures[n]), sendCount, MPI_DOUBLE, dst, 117, &(ures[(m + 1) * n]), recvCount, MPI_DOUBLE, src, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&(ures[m * n]), recvCount, MPI_DOUBLE, src, 0, &(ures[0]), sendCount, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 : 2; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL SENDRECV " << " NODES " << np << " Treads " << omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;
    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }
}
void testZeidelMPIompISend_IRecv(int n, int argc, char** argv)
{
    double t1 = MPI_Wtime();
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double h = L / (n - 1);
    lambda = (1 / h) * (1 / h);
    double* u;
    double* ures;
    double flag = 1;
    double* f;
    f = new double[np];
    double sflag = 1.;
    int m = (n - 2) / np;
    u = new double[n * (m + 2)];
    ures = new double[n * (m + 2)];
    double* R;
    R = new double[n * n];
    double* A;
    A = new double[n * n];
    if (myid == 0)
    {
        for (int i = 0; i < n * n; i++)
            R[i] = 1.00001;
        for (int i = 0; i < n; i++)
        {
            R[n * (n - 1) + i] = Granvh(i * h);
            R[i * n] = Granlf(i * h);
            R[i] = Grannz(i * h);
            R[i * n + n - 1] = Granpr(i * h);
        }
        for (int i = 1; i < np; i++)
            MPI_Send(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        for (int i = 0; i < n * (m + 2); i++)
            u[i] = R[i];
    }
    if (myid > 0)
        MPI_Recv(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n * (m + 2); i++)
        ures[i] = u[i];
    double t;
    double res = 1;
    int iter = 0;
    MPI_Request senddown0, recvdown0, sendup0, recvup0, senddown1, recvdown1, sendup1, recvup1;
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[m * n]), n, MPI_DOUBLE, i + 1, 117, MPI_COMM_WORLD, &senddown0);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(u[0]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvdown0);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(u[n]), n, MPI_DOUBLE, i - 1, 117, MPI_COMM_WORLD, &sendup0);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(u[(m + 1) * n]), n, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, &recvup0);
        }
    }
    for (int i = 0; i < np - 1; i++)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[m * n]), n, MPI_DOUBLE, i + 1, 116, MPI_COMM_WORLD, &senddown1);
        }
        if (myid == i + 1)
        {
            MPI_Recv_init(&(ures[0]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvdown1);
        }
    }
    for (int i = np - 1; i > 0; i--)
    {
        if (myid == i)
        {
            MPI_Send_init(&(ures[n]), n, MPI_DOUBLE, i - 1, 116, MPI_COMM_WORLD, &sendup1);
        }
        if (myid == i - 1)
        {
            MPI_Recv_init(&(ures[(m + 1) * n]), n, MPI_DOUBLE, i, 116, MPI_COMM_WORLD, &recvup1);
        }
    }
    iter = 0;
    
    while (sflag > eps)
    {
        if (iter % 2 == 0)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
#pragma omp parallel for
        for (int j = 2; j < m; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 2 : 1; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + u[i + 1 + j * n] + u[i - 1 + j * n] + u[i + (j + 1) * n] + u[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 0)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
#pragma omp parallel for
        for (int i = ((1 + m * myid) % 2 == 0) ? 1 : 2; i < n - 1; i += 2)
            ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + u[i + 1 + n] + u[i - 1 + n] + u[i + (1 + 1) * n] + u[i + (1 - 1) * n]) / (4 + lambda * h * h);
#pragma omp parallel for
        for (int i = ((m + m * myid) % 2 == 0) ? 1 : 2; i < n - 1; i += 2)
            ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + u[i + 1 + m * n] + u[i - 1 + m * n] + u[i + (m + 1) * n] + u[i + (m - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 1)
        {

            if (myid < np - 1)
                MPI_Start(&senddown0);
            if (myid > 0)
                MPI_Start(&sendup0);
            if (myid < np - 1)
                MPI_Start(&recvup0);
            if (myid > 0)
                MPI_Start(&recvdown0);
        }
        else
        {
            if (myid < np - 1)
                MPI_Start(&senddown1);
            if (myid > 0)
                MPI_Start(&sendup1);
            if (myid < np - 1)
                MPI_Start(&recvup1);
            if (myid > 0)
                MPI_Start(&recvdown1);
        }
#pragma omp parallel for
        for (int j = 2; j < m; j++)
            for (int i = ((j + m * myid) % 2 == 1) ? 1 : 2; i < n - 1; i += 2)
                ures[i + j * n] = (PR(i * h, (j + myid * m) * h) * h * h + ures[i + 1 + j * n] + ures[i - 1 + j * n] + ures[i + (j + 1) * n] + ures[i + (j - 1) * n]) / (4 + lambda * h * h);
        if (iter % 2 == 1)
        {
            if (myid < np - 1)
                MPI_Wait(&recvup0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown0, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup0, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown0, MPI_STATUS_IGNORE);
        }
        else
        {
            if (myid < np - 1)
                MPI_Wait(&recvup1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&recvdown1, MPI_STATUS_IGNORE);
            if (myid > 0)
                MPI_Wait(&sendup1, MPI_STATUS_IGNORE);
            if (myid < np - 1)
                MPI_Wait(&senddown1, MPI_STATUS_IGNORE);
        }
#pragma omp parallel for
        for (int i = ((1 + m * myid) % 2 == 0) ? 2 : 1; i < n - 1; i += 2)
            ures[i + n] = (PR(i * h, (1 + myid * m) * h) * h * h + ures[i + 1 + n] + ures[i - 1 + n] + ures[i + (1 + 1) * n] + ures[i + (1 - 1) * n]) / (4 + lambda * h * h);
#pragma omp parallel for
        for (int i = ((m + m * myid) % 2 == 0) ? 2 : 1; i < n - 1; i += 2)
            ures[i + m * n] = (PR(i * h, (m + myid * m) * h) * h * h + ures[i + 1 + m * n] + ures[i - 1 + m * n] + ures[i + (m + 1) * n] + ures[i + (m - 1) * n]) / (4 + lambda * h * h);
        iter++;
        res = 0;
#pragma omp parallel for
        for (int j = 1; j < m + 1; j++)
            for (int i = 1; i < n - 1; i++)
                res += (u[i + j * n] - ures[i + j * n]) * (u[i + j * n] - ures[i + j * n]);
        res = std::sqrt(res);
        flag = res;
        if (myid > 0)
            MPI_Send(&flag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
        if (myid == 0)
        {
            f[0] = flag;
            sflag = 0;
            for (int i = 1; i < np; i++)
                MPI_Recv(&f[i], 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < np; i++)
                sflag += f[i];
            for (int i = 1; i < np; i++)
                MPI_Send(&sflag, 1, MPI_DOUBLE, i, 117, MPI_COMM_WORLD);
        }
        if (myid > 0)
            MPI_Recv(&sflag, 1, MPI_DOUBLE, 0, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(u, ures);
    }
    
    if (myid > 0)
        MPI_Send(&(u[0]), n * (m + 2), MPI_DOUBLE, 0, 117, MPI_COMM_WORLD);
    if (myid == 0)
    {
        for (int i = 0; i < n * (m + 2); i++)
            R[i] = u[i];
        for (int i = 1; i < np; i++)
            MPI_Recv(&(R[(i * m) * n]), n * (m + 2), MPI_DOUBLE, i, 117, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n < 20)
        {
            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << R[i * n + j] << ' ';
                }
                std::cout << std::endl;
            }

            std::cout << "______________________" << std::endl;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    std::cout << F(i * h, j * h) << ' ';
                }
                std::cout << std::endl;
            }
            std::cout << "______________________" << std::endl;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = F(i * h, j * h);
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ZEIDEL ISEND_IRECV " << " NODES " << np <<" Treads "<<omp_get_max_threads() << " matrix size " << n - 2 << std::endl;
        std::cout << "Norm C " << NormMatr1(R, A, n, n) << std::endl;
        std::cout << "Count iter " << iter << std::endl;

    }
    delete[] R;
    delete[] A;
    delete[] u;
    delete[] ures;
    double t2 = MPI_Wtime();
    if (myid == 0)
    {
        std::cout << "Time " << t2 - t1 << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }

}
int main(int argc,char** argv)
{
    MPI_Init(&argc, &argv);
    int myid, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int k =500;
    int n = k-k%np+2;
    //testJacobiMPISend_Recv(n,argc,argv);
    //testJacobiMPISR(n,argc,argv);
    //testJacobiMPIISend_IRecv(n,argc,argv);
    //testZeidelMPISend_Recv(n, argc, argv);
    //testZeidelMPISR(n, argc, argv);
   // testZeidelMPIISend_IRecv(n, argc, argv);
    testZeidel_if(n, myid);
    testZeidel(n,myid);
    testJacobi(n, myid);
    //for (int i = 2; i < 5; i++)
    //{
    //        omp_set_num_threads(i);
    //        if (myid == 0)
    //        {
    //            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //            std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    //        }
    //        MPI_Barrier(MPI_COMM_WORLD);
    //        testJacobiMPIompSend_Recv(n, argc, argv);
    //        testJacobiMPIompSR(n, argc, argv);
    //        testJacobiMPIompISend_IRecv(n, argc, argv);
    //        testZeidelMPIompSend_Recv(n, argc, argv);
    //        testZeidelMPIompSR(n, argc, argv);
    //        testZeidelMPIompISend_IRecv(n, argc, argv);
    //        //testompJacobi_Zeidel(n, myid);
    //        MPI_Barrier(MPI_COMM_WORLD);
    //        if (myid == 0)
    //            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //        std::cout << std::endl;
    //        std::cout << std::endl;
    //        std::cout << std::endl;
    //}
    //mpitest(argc,argv);
   // std::cout << "Hello World!\n";
   MPI_Finalize();
    return 0;
}
