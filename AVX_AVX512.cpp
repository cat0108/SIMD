#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
using namespace std;
alignas(16) float gdata[10000][10000];//���ж������
alignas(16) float gdata2[10000][10000];
float gdata1[10000][10000];
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j]= gdata2[i][j];
		}
	}
	//����һ�мӵ����У��ڶ��мӵ����У��Դ����ƣ���֤��Ԫʱ�������������������0
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
			}
		}
	}

}

void Normal_alg(int N)
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}

//����avx
void Par_alg_avx(int n)
{
	int i, j, k;
	__m256 r0, r1, r2, r3;//8·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[8] = { gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k] };
		r0 = _mm256_load_ps(temp);//�ڴ�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 8 <= n; j += 8)
		{
			r1 = _mm256_load_ps(gdata2[k] + j);
			r1 = _mm256_div_ps(r1, r0);//������������λ���
			_mm256_store_ps(gdata2[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��8�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
		}
		gdata2[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[8] = { gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k] };
			r0 = _mm256_load_ps(temp2);
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				r1 = _mm256_load_ps(gdata2[k] + j);
				r2 = _mm256_load_ps(gdata2[i] + j);
				r3 = _mm256_mul_ps(r0, r1);
				r2 = _mm256_sub_ps(r2, r3);
				_mm256_store_ps(gdata2[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata2[i][j] = gdata2[i][j] - (gdata2[i][k] * gdata2[k][j]);
			}
			gdata2[i][k] = 0;
		}
	}
}

//����avx512
void Par_alg_avx_512(int n)
{
	int i, j, k;
	__m512 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[16] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k], gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k], gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k], gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm512_load_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 16 <= n; j += 16)
		{
			r1 = _mm512_load_ps(gdata[k] + j);
			r1 = _mm512_div_ps(r1, r0);//������������λ���
			_mm512_store_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[16] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm512_load_ps(temp2);
			for (j = k + 1; j + 16 <= n; j += 16)
			{
				r1 = _mm512_load_ps(gdata[k] + j);
				r2 = _mm512_load_ps(gdata[i] + j);
				r3 = _mm512_mul_ps(r0, r1);
				r2 = _mm512_sub_ps(r2, r3);
				_mm512_store_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}




int main()
{
	LARGE_INTEGER fre, begin, end;
	double gettime;
	int n;
	cin >> n;
	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Initialize(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "intial time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Normal_alg(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "normal time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_avx(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "avx_Parallel time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_avx_512(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "avx512_Parallel time: " << gettime << " ms" << endl;
}