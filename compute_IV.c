#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<ctype.h>
#include<math.h>

#define PI 3.14159265358979323846
#define E 2.71828182845904523536

int BROKEN_TOL_COUNT = 0;

typedef struct{
    double time2mat;
    double strike;
    double price;
} option_t;
    

inline double cube(const double x){return x*x*x;}
inline double stdNormalCdf(const double x);
inline double priceCallOption(const double sigma, const double S, const double r, const double K, const double T);
inline double pricePutOption(const double sigma, const double S, const double r, const double K, const double T);
inline double computeVega(const double sigma, const double S, const double r, const double K, const double T);
inline double computeDelta(const double sigma, const double S, const double r, const double K, const double T);

int main(int argc, char *argv[]){
    

    if(argc != 4){
        printf("Usage: ./compute_IV coin:char side:char type:char\n");
        return -1;
    }

    char symbol = *argv[1];
    char side = *argv[2];
    char optionType = *argv[3];
    double  (*pricingFunction)(const double, const double, const double, const double, const double);


    char coin[4];
    switch (symbol)
    {
    case 'b':
        strcpy(coin, "btc");
        break;
    
    case 'e':
        strcpy(coin, "eth");
        break;
    
    case 'd':
        strcpy(coin, "dog");
        break;
    default:
        printf("Error parsing symbol: Symbol must be either: b/e/d (btc/eth/doge).\n");
        return -1;
    }

    char sideStr[4];
    switch (side)
    {
    case 'a':
        strcpy(sideStr, "ASK");
        break;
    
    case 'b':
        strcpy(sideStr, "BID");
        break;
    default:
        printf("Error parsing side: Side must be either: b/a (BID/ASK)");
        return -1;
        
    }
    
    
    switch (optionType)
    {
    case 'p':
        pricingFunction = pricePutOption;
        break;
    case 'c':
        pricingFunction = priceCallOption;
        break;
    default:
        printf("Error parsing type: type must be c/p (call/put)\n");
        return -1;
        break;
    }

    time_t currentTime;
    struct tm *localTime;

    time(&currentTime);
    localTime = localtime(&currentTime);

    // Format current date as string in yymmdd format
    char dateString[7]; // 6 characters for the date and 1 for null terminator
    strftime(dateString, sizeof(dateString), "%y%m%d", localTime);

    char filename[sizeof(dateString) + sizeof(optionType) + sizeof(sideStr) + sizeof(coin) + 7];

    strcpy(&filename[0], coin);
    filename[3] = '_';
    strcpy(&filename[4], sideStr);
    filename[7] = '_';
    strcpy(&filename[8], dateString);
    filename[14] = '_';
    filename[15] = toupper(optionType);
    strcpy(&filename[16], ".bin");

    FILE * optionFile = fopen(filename, "rb");

    if(optionFile == NULL){
        printf("Error: Something happend reading option file. Does not exist?\n");
        return -1;
    }

    option_t *options = NULL;
    int n_options = 0;
    int read_counter;

    while (1)
    {
        options = realloc(options, (n_options + 1)*sizeof(option_t));

        read_counter = fread(&options[n_options], sizeof(option_t), 1, optionFile);
        if(read_counter != 1){
            break; //EOF
        }
        n_options++;
    }
    fclose(optionFile);

    //Read underlying price
    double underlyingPrice;

    FILE * underlyingFile;
    char underlyingFileName[sizeof(coin) + 4];
    strcpy(underlyingFileName, coin);
    strcpy(&underlyingFileName[3], ".bin");
    underlyingFile = fopen(underlyingFileName, "rb");
    read_counter = fread(&underlyingPrice, sizeof(double), 1, underlyingFile);
    if(read_counter != 1){
        printf("Error: Something went wrong reading underlying price.\n");
        return -1;
    }
    fclose(underlyingFile);

    double *prices = (double *)malloc(n_options * sizeof(double));
    double *strikes = (double *)malloc(n_options * sizeof(double));
    double *maturties = (double *)malloc(n_options * sizeof(double));
    double *sigmas = (double *)malloc(n_options * sizeof(double));
    double *deltas = (double *)malloc(n_options * sizeof(double));
    //Copy data
    for(int i=0; i<n_options;i++){
        maturties[i] = options[i].time2mat;
        strikes[i] =  options[i].strike;
        prices[i] = options[i].price;
        
    }
    

    double r = 0.04; //This should be fixed to account for some model
    double sigma;
    double K;
    double S = underlyingPrice;
    double T;
    double optionStar;
    double tol=1e-4;
    int append_index = 0;
    int k;
    double sigmaGuesses[5] = {0.1, 0.25, 0.5, 1.0, 1.5};

    
    int success = 0;
            for(int i=0;i<n_options;i++){
                K = strikes[i];
                T = maturties[i];
                optionStar = prices[i];
            
            
            for(k=0;k<5;k++){
                sigma = sigmaGuesses[k];
          //Do  Newton iterations
                for(int j=0; j<250;j++){
                        sigma -= (pricingFunction(sigma, S, r, K, T) - optionStar) / computeVega(sigma, S, r, K, T);
                    }
                        if((abs(optionStar - pricingFunction(sigma, S, r, K, T)) > tol)){
                            BROKEN_TOL_COUNT++;
                       }
                        else{
                            success++;
                            sigmas[append_index] = sigma;
                            strikes[append_index] =  K;
                            deltas[append_index] = computeDelta(sigma, S, r, K, T);
                            append_index++;
                            break;
                        }



            }


    }
    printf("Succesfully computed IV %d times\n", success);
    //Write to output file
    FILE * outfile;
    char out_filename[64];
    snprintf(out_filename, sizeof(out_filename), "out_%c.bin", optionType);
    outfile = fopen(out_filename, "wb");
    for(int i=0; i<append_index;i++){
        fwrite(&maturties[i], sizeof(double), 1, outfile);
        fwrite(&strikes[i], sizeof(double), 1, outfile); //We are using this for moneyness here
        fwrite(&sigmas[i], sizeof(double), 1, outfile);
        fwrite(&deltas[i], sizeof(double), 1, outfile);

    }
    fclose(outfile);

    free(prices);
    free(strikes);
    free(maturties);
    free(sigmas);
    free(deltas);
    return 0;
}



double erf(const double x){
    double x_sq = x*x;
    double x_cube = x_sq*x;

    return 2.0/(15.0 * sqrt(PI)) * (49140*x + 3570*x_cube + 739*x_cube*x_sq)/(165*x*x_cube + 1330*x_sq + 3276);
}

double stdNormalCdf(const double x){
    return 0.5*(1+erf(x/sqrt(2)));
}

double pricePutOption(const double sigma, const double S, const double r, const double K, const double T){
    double sqrtT = sqrt(T);
    
    double d1 = (log(S/K) + (r + sigma*sigma*0.5)*T)/(sigma*sqrtT);
    double d2 = d1 - sigma*sqrtT;

    return  K*pow(E, -r*T) * stdNormalCdf(-d2) - S*stdNormalCdf(-d1);
}

double priceCallOption(const double sigma, const double S, const double r, const double K, const double T){
    double sqrtT = sqrt(T);
    
    double d1 = (log(S/K) + (r + sigma*sigma*0.5)*T)/(sigma*sqrtT);
    double d2 = d1 - sigma*sqrtT;

    return S*stdNormalCdf(d1) - K*pow(E, -r*T) * stdNormalCdf(d2);
}
double computeVega(const double sigma, const double S, const double r, const double K, const double T){
    double sqrtT = sqrt(T);
    double d1 = (log(S/K) + (r + sigma*sigma*0.5)*T)/(sigma*sqrtT);

    return S*pow(E, d1)*sqrtT;
}

double computeDelta(const double sigma, const double S, const double r, const double K, const double T){
    return stdNormalCdf((log(S/K) + (r + sigma*sigma*0.5)*T)/(sigma*sqrt(T)));
}

 
