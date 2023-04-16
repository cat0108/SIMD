#pragma once

class BitMap {
	unsigned int* bits = NULL;//�ܵĿ�������
	int size;//�ܵĿ�������ĸ���
	int range;//��Ԫ�ظ���
public:
	BitMap() { };
	BitMap(int range)
	{
		this->range = range;
		size =(int)( range / 32 + 1);
		bits = new unsigned int[size];
	}
	~BitMap()
	{
		delete bits;
	}
	void add(int indata)
	{
		int index = (int)(indata / 32);
		int bit_index = indata % 32;
		bits[index] |= 1 << bit_index;//��λͼ��ĳһλ��Ϊ1
	}
	void remove(int indata) 
	{
		int index = (int)(indata / 32);
		int bit_index = indata % 32;
		bits[index] &= ~(1 << bit_index);//����λ��Ϊ0
	}
	bool Find(int indata)
	{
		int index = int (indata / 32);
		int bit_index = indata % 32;
		return (bits[index] >> bit_index) & 1;
	}
	void initialize()
	{
		for (int i = 0; i < size; i++)
			bits[i] = 0;
	}

	bool judge_zero()
	{
		bool flag = 1;
		for (int i = 0; i < size; i++)//���Բ��л��ж��Ƿ�ȫΪ0
			if (bits[i] != 0)
				flag = 0;
		return flag;
	}

	unsigned int  find_max()
	{
		unsigned int max1=0;
		for (int i = 0; i < range; i++)
		{
			if (Find(i) == 1)
				max1 = i;
		}
		return max1;
	}
	int Getsize() { return size; }
	unsigned int* Getbits() { return bits; }
};