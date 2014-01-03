#include <algorithm>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")

using namespace std;

typedef long long ll; 
#define all(c) (c).begin(),(c).end() 
typedef pair<int,int> Pii;
typedef vector<int> Vi;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

#define THREADS 256
#define MEGA 1307674368000LL

#define NUM_ELEMENTS 13//have not tested beyond 15!, but should be able to handle if you adjust THREADS, blockSize and MEGA
#define DO_TEST 1

const int blockSize=4096;

inline int get_adj_size(const long long num_elem){
	double p=double(num_elem)/double(MEGA);
	if(p>0.8)return 5;
	else if(p>0.6)return 4;
	else if(p>0.4)return 3;
	else if(p>0.2)return 2;
	else
		return 1;
}
inline int get_dynamic_block_size(const int adj_size,const int blkSize){
	return (1<<(adj_size-1))*blkSize;
}

//for testing full version and verification of answers on host
const long long H_F[16]={1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600LL,6227020800LL,87178291200LL,1307674368000LL};
const int H_depend[15]={4,8,32,0,1,512,16,2,64,128,256,1024,3,35,129};//acts as dependency bitmask
const int h_val[15]={33,438,19,277,449,129,40,5,22,127,61,7,3,111,1};

//GPU info for full version
__constant__ long long D_F[16]={1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600LL,6227020800LL,87178291200LL,1307674368000LL};
__constant__ int D_depend[15]={4,8,32,0,1,512,16,2,64,128,256,1024,3,35,129};//acts as dependency bitmask
__constant__ int d_val[15]={33,438,19,277,449,129,40,5,22,127,61,7,3,111,1};//value associated with that index (only if the dependencies are met, otherwise adds value of 0)

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);
void _cpu_derive2(const long long num, vector<int> &V,int digits);
void _printV(const int nume,const vector<int> &AA);
int _add_up(const int nume,const vector<int> &AA);
long long fact_val(long long cur){return cur<=1LL ? 1LL:(cur*fact_val(cur-1LL));}

//basic CUDA GPU kernel which calcuates the permIdx'th permutation of (digits) digits
template<int blockWork>
__global__ void _gpu_perm_basic(const int digits);//this goes through the permutations with any type of analysis
__global__ void _gpu_perm_basic_last_step(const long long bound,const int digits,const long long rem_start);

//generates all permutations AND evaluates the current permutation using __constant__ memory and stores the best result (along with the respective permutation)
template<int blockWork>
__global__ void _gpu_perm(int* __restrict__ ans_val,int2* __restrict__ perm_val,const int digits);
__global__ void _gpu_perm_last_step(int* __restrict__ ans_val,int2* __restrict__ perm_val,const long long bound,
	const int digits,const long long rem_start,const int num_blox);

//NOTE: This 13! and up version will only work on GPUs with compute capability of 3.0 or higher (GTX 660 and up)

//Python est lent!

int main(){
        char ch;
        srand(time(NULL));

		const bool test_raw=true;//for testing of raw version or full version with permutation evaluation
    
		int compute_capability=0;
		cudaDeviceProp deviceProp;
		cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		string ss= (deviceProp.major>=3 && deviceProp.minor>=0) ? "Capable!\n":"Not Sufficient compute capability!\n";
		cout<<ss;
		err=cudaDeviceReset();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		//Windows timer stuff declarations
        DWORD startTime=0,endTime=0,GPUtime=0;
		UINT wTimerRes = 0;
		bool init=false;
		//

        if(DO_TEST && deviceProp.major>=3){
				cout<<"Starting GPU testing:\n";

                const long long result_space=H_F[NUM_ELEMENTS];
				const int adj_size=get_adj_size(result_space);
				const int temp_blocks_sz=get_dynamic_block_size(adj_size,blockSize);
				const int num_blx=int(result_space/long long(temp_blocks_sz));

				const long long rem_start=result_space-(result_space-long long(num_blx)*long long(temp_blocks_sz));

				if(!test_raw){//test with permutation evaluation, optimization,scan,reduction etc..

					int GPUans=0;
					long long GPU_perm_number=0LL;

					cout<<"\nTesting full version.\n";
					int2 perm_mask_split={0};
					int *ans_val;
					int2 *perm_val;

					const unsigned int num_bytes_ans=num_blx*sizeof(int);
					const unsigned int num_bytes_perm=num_blx*sizeof(int2);
					err=cudaMalloc((void **)&ans_val,num_bytes_ans);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
					err=cudaMalloc((void **)&perm_val,num_bytes_perm);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
					err=cudaMemset(ans_val,0,num_blx*sizeof(int));
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

               
					wTimerRes = 0;
					init = InitMMTimer(wTimerRes);
					startTime = timeGetTime();

					if(adj_size==1){
						_gpu_perm<blockSize><<<num_blx,THREADS>>>(ans_val,perm_val,NUM_ELEMENTS);
					}else if(adj_size==2){
						_gpu_perm<blockSize*2><<<num_blx,THREADS>>>(ans_val,perm_val,NUM_ELEMENTS);
					}else if(adj_size==3){
						_gpu_perm<blockSize*4><<<num_blx,THREADS>>>(ans_val,perm_val,NUM_ELEMENTS);
					}else if(adj_size==4){
						_gpu_perm<blockSize*8><<<num_blx,THREADS>>>(ans_val,perm_val,NUM_ELEMENTS);
					}else{
						_gpu_perm<blockSize*16><<<num_blx,THREADS>>>(ans_val,perm_val,NUM_ELEMENTS);
					}
					err = cudaThreadSynchronize();
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

					_gpu_perm_last_step<<<1,THREADS>>>(ans_val,perm_val,result_space,NUM_ELEMENTS,rem_start,num_blx);
					err = cudaThreadSynchronize();
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

					err=cudaMemcpy(&GPUans,ans_val,sizeof(int),_DTH);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
			
					err=cudaMemcpy(&perm_mask_split,perm_val,sizeof(int2),_DTH);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
			

					endTime = timeGetTime();
					GPUtime=endTime-startTime;
					cout<<"GPU timing: "<<float(GPUtime)/1000.0f<<" seconds.\n";
					cout<<"GPU answer is: "<<GPUans<<'\n';
					GPU_perm_number=*reinterpret_cast<long long *>(&perm_mask_split);

					DestroyMMTimer(wTimerRes, init);

					err=cudaFree(ans_val);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
					err=cudaFree(perm_val);
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
					cout<<"\nThe evaluation had ( n!*(4+2*n+n^2)) steps, which is apx. "<<ll(H_F[NUM_ELEMENTS])*ll(4+2*NUM_ELEMENTS+(NUM_ELEMENTS*NUM_ELEMENTS))<<" iterations.\n";

                                       
					vector<int> V(NUM_ELEMENTS,0);
                
					_cpu_derive2(GPU_perm_number,V,NUM_ELEMENTS);
                
					//int ts=_add_up(NUM_ELEMENTS,V);//this should match answer
					_printV(NUM_ELEMENTS,V);

				}else{//just test the array permutation generation by itself

					cout<<"\nTesting raw permutation version.\n";

					wTimerRes = 0;
					init = InitMMTimer(wTimerRes);
					startTime = timeGetTime();

					if(adj_size==1){
						_gpu_perm_basic<blockSize><<<num_blx,THREADS>>>(NUM_ELEMENTS);
					}else if(adj_size==2){
						_gpu_perm_basic<blockSize*2><<<num_blx,THREADS>>>(NUM_ELEMENTS);
					}else if(adj_size==3){
						_gpu_perm_basic<blockSize*4><<<num_blx,THREADS>>>(NUM_ELEMENTS);
					}else if(adj_size==4){
						_gpu_perm_basic<blockSize*8><<<num_blx,THREADS>>>(NUM_ELEMENTS);
					}else{
						_gpu_perm_basic<blockSize*16><<<num_blx,THREADS>>>(NUM_ELEMENTS);
					}
					err = cudaThreadSynchronize();
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

					_gpu_perm_basic_last_step<<<1,THREADS>>>(result_space,NUM_ELEMENTS,rem_start);
					err = cudaThreadSynchronize();
					if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

					endTime = timeGetTime();
					GPUtime=endTime-startTime;
					cout<<"GPU timing for "<<NUM_ELEMENTS <<"!: "<<double(GPUtime)/1000.0f<<" seconds.\n";
					DestroyMMTimer(wTimerRes, init);
					cout <<H_F[NUM_ELEMENTS]<<" permutations generated, took apx "<<long long(NUM_ELEMENTS*NUM_ELEMENTS)*H_F[NUM_ELEMENTS]<<" iterations/calc on gpu bitches!\n";

				}
        }

        cin>>ch;
        return 0;
}

bool InitMMTimer(UINT wTimerRes){
        TIMECAPS tc;
        if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
        wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
        timeBeginPeriod(wTimerRes); 
        return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
        if(init)
			timeEndPeriod(wTimerRes);
}

void _cpu_derive2(const long long num, vector<int> &V,const int digits){
        long long tnum=num,c;
        int B[19]={0};
        for(int i=0;i<digits;i++){B[i]=i;}
        for(int d=digits-1;d>=0;d--){
                c=long long(d);                 
                while(c*H_F[d]>tnum){c--;} 
                if(d==digits-1){
					V[d]=B[int(c)];
					B[int(c)]=-1;
                }else{
                        int cc=0;
                        for(int ii=0;ii<digits;ii++){
                                if(B[ii]!=-1){
                                        if(cc==int(c)){
                                                V[d]=B[ii];
                                                B[ii]=-1;
                                                break;
                                        }else{
                                                cc++;
                                        }
                                }
                        }        
                }
                tnum-=c*H_F[d];
        }

}

void _printV(const int nume,const vector<int> &AA){
        int sofar=0,d=0;
        cout<<"\nHere is a permutation associated with the best result. If there is a bracket followed by a 0 that means the dependencies which were necessary to include that value were not met.\n";
        for(int i=0;i<nume;i++){

           if( (d&H_depend[AA[i]])==H_depend[AA[i]]){
				sofar+=h_val[AA[i]];
                cout <<"["<<AA[i]<<"]= "<<h_val[AA[i]]<<',';
           }else
               cout<<"["<<AA[i]<<"]=0, ";

           d|=(1<<AA[i]);
		}

        cout<<"\nGPU total="<<sofar<<'\n';
}

int _add_up(const int nume,const vector<int> &AA){
        int ans=0,d=0;
        for(int i=0;i<nume;i++){
			if( (d&H_depend[AA[i]])==H_depend[AA[i]])ans+=h_val[AA[i]];
			d|=(1<<AA[i]);
        }

        return ans;
}

//this version has not built in evaluation step, just goes through all permutations in local GPU memory space
template<int blockWork>
__global__ void _gpu_perm_basic(const int digits){

		const long long offset=long long(threadIdx.x)+long long(blockIdx.x)*long long(blockWork);
		const int reps=blockWork>>8;//assuming 256 threads

        int A[NUM_ELEMENTS],B[NUM_ELEMENTS],d,cc,ii,idx;
		long long tnum,c;
		for(ii=0;ii<reps;ii++){
			tnum=offset+long long(ii*THREADS);
			idx=0;
			while(idx<digits){
					B[idx]=idx;
					idx++;
			}
			for(d=digits-1;d>=0;d--){
					c=long long(d);
					while(c*D_F[d]>tnum){c--;}
					if(d==digits-1){
							A[d]=B[int(c)];
							B[int(c)]=-1;

					}else{
							idx=0;cc=0;
							while(idx<digits){
									if(B[idx]!=-1){
											if(cc==int(c)){
													A[d]=B[idx];
													B[idx]=-1;
													break;
											}else
													cc++;
									}
									idx++;
							}
					}
					tnum-=c*D_F[d];
			}
			//now you have the unique permutation of indexes stored in thread local array A[], do what you want from here with that permutation of indexes
		}
}

__global__ void _gpu_perm_basic_last_step(const long long bound,const int digits,const long long rem_start){

	const long long offset=long long(threadIdx.x)+rem_start;
	
    int A[NUM_ELEMENTS],B[NUM_ELEMENTS],d,cc,ii,idx;
	long long tnum,c,adj=0LL;
	for(;(offset+adj)<bound;ii++){
		tnum=offset+adj;
		idx=0;
		while(idx<digits){
			B[idx]=idx;
			idx++;
		}
		for(d=digits-1;d>=0;d--){
			c=long long(d);
			while(c*D_F[d]>tnum){c--;}
			if(d==digits-1){
				A[d]=B[int(c)];
				B[int(c)]=-1;

			}else{
				idx=0;cc=0;
				while(idx<digits){
					if(B[idx]!=-1){
						if(cc==int(c)){
							A[d]=B[idx];
							B[idx]=-1;
							break;
						}else
							cc++;
					}
					idx++;
				}
			}
			tnum-=c*D_F[d];
		}
		adj=(long long(ii)<<8LL);

	}
	//done, and evaluation step of current permutation in array A would take place after this line in kernel
}

//for full permutations using __constant__ array to store dependencies and value info

template<int blockWork>
__global__ void _gpu_perm(int* __restrict__ ans_val,int2* __restrict__ perm_val,const int digits){

	const long long offset=long long(threadIdx.x)+long long(blockIdx.x)*long long(blockWork);
	const int reps=blockWork>>8;//NOTE: Hardcoded for 256 threads per block, 8 warps
	const int warpIndex = threadIdx.x%32;

	__shared__ int blk_best[8];
	__shared__ int2 mask_val[8];

    int A[NUM_ELEMENTS],B[NUM_ELEMENTS],d,cc,ii,value=-(1<<29),idx;
	int2 mask_as_int2,t2;
	long long tnum,c;

	for(ii=0;ii<reps;ii++){
		tnum=offset+long long(ii*THREADS);
		idx=0;
		while(idx<digits){
			B[idx]=idx;
			idx++;
		}
		for(d=digits-1;d>=0;d--){
			c=long long(d);
			while(c*D_F[d]>tnum){c--;}
			if(d==digits-1){
				A[d]=B[int(c)];
				B[int(c)]=-1;

			}else{
				idx=0;cc=0;
				while(idx<digits){
					if(B[idx]!=-1){
						if(cc==int(c)){
							A[d]=B[idx];
							B[idx]=-1;
							break;
						}else
							cc++;
						}
					idx++;
				}
			}
			tnum-=c*D_F[d];
		}
		d=0;
		cc=0;
		for(idx=0;idx<digits;idx++){
			if((d&D_depend[A[idx]])==D_depend[A[idx]]){//check to see if dependencies have been met to add this value
				cc+=d_val[A[idx]];
			}
            d|=(1<<A[idx]);//mark as having been 'seen'
		}
		if(cc>value){
			value=cc;
			tnum=offset+long long(ii*THREADS);
			mask_as_int2=*reinterpret_cast<int2 *>(&tnum);
		}
	}

	tnum=__shfl(value,warpIndex+16);
	t2.x=__shfl(mask_as_int2.x,warpIndex+16);
	t2.y=__shfl(mask_as_int2.y,warpIndex+16);
	if(tnum>value){
		value=tnum;
		mask_as_int2=t2;
	}
	tnum=__shfl(value,warpIndex+8);
	t2.x=__shfl(mask_as_int2.x,warpIndex+8);
	t2.y=__shfl(mask_as_int2.y,warpIndex+8);
	if(tnum>value){
		value=tnum;
		mask_as_int2=t2;
	}
	tnum=__shfl(value,warpIndex+4);
	t2.x=__shfl(mask_as_int2.x,warpIndex+4);
	t2.y=__shfl(mask_as_int2.y,warpIndex+4);
	if(tnum>value){
		value=tnum;
		mask_as_int2=t2;
	}
	tnum=__shfl(value,warpIndex+2);
	t2.x=__shfl(mask_as_int2.x,warpIndex+2);
	t2.y=__shfl(mask_as_int2.y,warpIndex+2);
	if(tnum>value){
		value=tnum;
		mask_as_int2=t2;
	}
	tnum=__shfl(value,warpIndex+1);
	t2.x=__shfl(mask_as_int2.x,warpIndex+1);
	t2.y=__shfl(mask_as_int2.y,warpIndex+1);
	if(tnum>value){
		value=tnum;
		mask_as_int2=t2;
	}

	if(threadIdx.x%32==0){
		blk_best[threadIdx.x>>5]=value;
		mask_val[threadIdx.x>>5]=mask_as_int2;
	}
	__syncthreads();

	if(threadIdx.x==0){
		tnum=blk_best[0];
		t2=mask_val[0];
		if(blk_best[1]>tnum){
			tnum=blk_best[1];
			t2=mask_val[1];
		}
		if(blk_best[2]>tnum){
			tnum=blk_best[2];
			t2=mask_val[2];
		}
		if(blk_best[3]>tnum){
			tnum=blk_best[3];
			t2=mask_val[3];
		}
		if(blk_best[4]>tnum){
			tnum=blk_best[4];
			t2=mask_val[4];
		}
		if(blk_best[5]>tnum){
			tnum=blk_best[5];
			t2=mask_val[5];
		}
		if(blk_best[6]>tnum){
			tnum=blk_best[6];
			t2=mask_val[6];
		}
		if(blk_best[7]>tnum){
			tnum=blk_best[7];
			t2=mask_val[7];
		}
		ans_val[blockIdx.x]=tnum;
		perm_val[blockIdx.x]=t2;
	}
}


__global__ void _gpu_perm_last_step(int* __restrict__ ans_val,int2* __restrict__ perm_val,const long long bound,
	const int digits,const long long rem_start,const int num_blox){

		const long long offset=long long(threadIdx.x)+rem_start;
		const int warpIndex = threadIdx.x%32;

		__shared__ int blk_best[8];
		__shared__ int2 mask_val[8];
	
		int A[NUM_ELEMENTS],B[NUM_ELEMENTS],d,cc,ii=1,value=-(1<<29),idx;
		int2 mask_as_int2,t2;
		long long tnum,c,adj=0LL;
		for(;(offset+adj)<bound;ii++){
			tnum=offset+adj;
			idx=0;
			while(idx<digits){
				B[idx]=idx;
				idx++;
			}
			for(d=digits-1;d>=0;d--){
				c=long long(d);
				while(c*D_F[d]>tnum){c--;}
				if(d==digits-1){
					A[d]=B[int(c)];
					B[int(c)]=-1;

				}else{
					idx=0;cc=0;
					while(idx<digits){
						if(B[idx]!=-1){
							if(cc==int(c)){
								A[d]=B[idx];
								B[idx]=-1;
								break;
							}else
								cc++;
							}
						idx++;
					}
				}
				tnum-=c*D_F[d];
			}
			d=0;
			cc=0;
			for(idx=0;idx<digits;idx++){
				if((d&D_depend[A[idx]])==D_depend[A[idx]]){//check to see if dependencies have been met to add this value
					cc+=d_val[A[idx]];
				}
				d|=(1<<A[idx]);//mark as having been 'seen'
			}
			if(cc>value){
				value=cc;
				tnum=offset+adj;
				mask_as_int2=*reinterpret_cast<int2 *>(&tnum);
			}
			adj=(long long(ii)<<8LL);
		}
		adj=0LL;
		for(ii=1;(threadIdx.x+int(adj))<num_blox;ii++){
			idx=(threadIdx.x+int(adj));
			if(ans_val[idx]>value){
				value=ans_val[idx];
				mask_as_int2=perm_val[idx];
			}
			adj=(long long(ii)<<8LL);
		}

		tnum=__shfl(value,warpIndex+16);
		t2.x=__shfl(mask_as_int2.x,warpIndex+16);
		t2.y=__shfl(mask_as_int2.y,warpIndex+16);
		if(tnum>value){
			value=tnum;
			mask_as_int2=t2;
		}
		tnum=__shfl(value,warpIndex+8);
		t2.x=__shfl(mask_as_int2.x,warpIndex+8);
		t2.y=__shfl(mask_as_int2.y,warpIndex+8);
		if(tnum>value){
			value=tnum;
			mask_as_int2=t2;
		}
		tnum=__shfl(value,warpIndex+4);
		t2.x=__shfl(mask_as_int2.x,warpIndex+4);
		t2.y=__shfl(mask_as_int2.y,warpIndex+4);
		if(tnum>value){
			value=tnum;
			mask_as_int2=t2;
		}
		tnum=__shfl(value,warpIndex+2);
		t2.x=__shfl(mask_as_int2.x,warpIndex+2);
		t2.y=__shfl(mask_as_int2.y,warpIndex+2);
		if(tnum>value){
			value=tnum;
			mask_as_int2=t2;
		}
		tnum=__shfl(value,warpIndex+1);
		t2.x=__shfl(mask_as_int2.x,warpIndex+1);
		t2.y=__shfl(mask_as_int2.y,warpIndex+1);
		if(tnum>value){
			value=tnum;
			mask_as_int2=t2;
		}

		if(threadIdx.x%32==0){
			blk_best[threadIdx.x>>5]=value;
			mask_val[threadIdx.x>>5]=mask_as_int2;
		}
		__syncthreads();
		
		if(threadIdx.x==0){
			tnum=blk_best[0];
			t2=mask_val[0];
			if(blk_best[1]>tnum){
				tnum=blk_best[1];
				t2=mask_val[1];
			}
			if(blk_best[2]>tnum){
				tnum=blk_best[2];
				t2=mask_val[2];
			}
			if(blk_best[3]>tnum){
				tnum=blk_best[3];
				t2=mask_val[3];
			}
			if(blk_best[4]>tnum){
				tnum=blk_best[4];
				t2=mask_val[4];
			}
			if(blk_best[5]>tnum){
				tnum=blk_best[5];
				t2=mask_val[5];
			}
			if(blk_best[6]>tnum){
				tnum=blk_best[6];
				t2=mask_val[6];
			}
			if(blk_best[7]>tnum){
				tnum=blk_best[7];
				t2=mask_val[7];
			}

			ans_val[0]=tnum;
			perm_val[0]=t2;

		}
}







