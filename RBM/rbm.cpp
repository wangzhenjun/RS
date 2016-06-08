#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <cstdio>
#include <cstring>

using namespace std;

const int num_of_user = 943;
const int num_of_movies = 1682;
const int num_of_rating = 5;
const int num_of_hidden = 200;
const int num_of_visible = 1682;

double uniform(double, double);
int binomial(double);

class RBM
{
public:
    int N;
    int n_visible;
    int n_hidden;
    int rating;
    double W[num_of_hidden][num_of_visible][num_of_rating];
    double hbias[num_of_hidden];
    double vbias[num_of_rating][num_of_visible];

    RBM(int, int, int, int);
    void contrastiveDivergence(int[][1682], double, int);
    void sample_h_given_v(int[][1682], double*, int*);
    double sigmoid(double);
    double Vtoh_sigm(int [][1682], double [][5], double);
    void gibbs_hvh(int*, double[][1682], int[][1682], double*, int*);
    double HtoV_sigm(int*, int, int, int);
    void sample_v_given_h(int* , double [][1682], int [][1682]);
    void reconstruct(int[][1682], double[][1682]);
};

void RBM::contrastiveDivergence(int train_data[][1682], double learning_rate, int k)
{
    //train_data 5 * 1682
    double ph_sigm_out[num_of_hidden]; // 10
    int ph_sample[num_of_hidden]; // 10
    double nv_sigm_outs[num_of_rating][num_of_visible]; // 5 * 1682
    int nv_samples[num_of_rating][num_of_visible]; // 5 * 1692
    double nh_sigm_outs[num_of_hidden]; // 10
    int nh_samples[num_of_hidden]; // 10

    sample_h_given_v(train_data, ph_sigm_out, ph_sample);

    for (int i = 0; i < k; ++i)
    {
        if (i == 0)
            gibbs_hvh(ph_sample, nv_sigm_outs, nv_samples, nh_sigm_outs, nh_samples);
        else
            gibbs_hvh(nh_samples, nv_sigm_outs, nv_samples, nh_sigm_outs, nh_samples);
    }

    for (int i = 0; i < n_hidden; ++i)
    {
        for (int j = 0; j < n_visible; ++j)
        {
            for (int kk = 0; kk < rating; ++kk)
            {
                W[i][j][kk] += learning_rate * (ph_sigm_out[i] * train_data[kk][j] - nh_sigm_outs[i] * nv_samples[kk][j]);
            }
        }
        hbias[i] += learning_rate * (ph_sigm_out[i] - nh_sigm_outs[i]) ;
    }

    for (int i = 0; i < rating; ++i)
    {
        for (int j = 0; j < n_visible; ++j)
        {
            vbias[i][j] += learning_rate * (train_data[i][j] - nv_samples[i][j]) ;
        }
    }
}

void RBM::gibbs_hvh(int* ph_sample, double nv_sigm_outs[][1682], int nv_samples[][1682], double* nh_sigm_outs, int* nh_samples)
{
    sample_v_given_h(ph_sample, nv_sigm_outs, nv_samples);
    sample_h_given_v(nv_samples, nh_sigm_outs, nh_samples);
}

void RBM::sample_h_given_v(int train_data[][1682], double* ph_sigm_out, int* ph_sample)
{
    for (int i = 0; i < n_hidden; ++i)
    {
        ph_sigm_out[i] = Vtoh_sigm(train_data, W[i], hbias[i]);
        ph_sample[i] = binomial(ph_sigm_out[i]);
    }
}

void RBM::sample_v_given_h(int* h0_sample, double nv_sigm_outs[][1682], int nv_samples[][1682])
{
    for (int i = 0; i < rating; ++i)
    {
        for (int j = 0; j < n_visible; ++j)
        {
            nv_sigm_outs[i][j] = HtoV_sigm(h0_sample, j, vbias[i][j], i);
            nv_samples[i][j] = binomial(nv_sigm_outs[i][j]);
        }
    }
}

double RBM::HtoV_sigm(int* h0_sample, int i, int vbias, int kk)
{
    double temp = 0;
    for (int j = 0; j < n_hidden; ++j)
    {
        temp += W[j][i][kk] * h0_sample[j];
    }
    temp += vbias;
    return sigmoid(temp);
}

double RBM::Vtoh_sigm(int train_data[][1682], double W[][5], double hbias)
{
    double temp = 0.0;
    for (int i = 0; i < rating; ++i)
    {
        for (int j = 0; j < n_visible; ++j)
            temp += W[j][i] * train_data[i][j];
    }
    temp += hbias;
    return sigmoid(temp);
}

double RBM::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

RBM::RBM(int train_N, int n_v, int n_h, int rt)
{
    N = train_N;
    n_visible = num_of_visible;
    n_hidden = num_of_hidden;
    rating = num_of_rating;

    double a = 1.0 / n_visible;
    for (int i = 0; i < n_hidden; ++i)
        for (int j = 0; j < n_visible; ++j)
            for (int k = 0; k < rating; ++k)
                W[i][j][k] = uniform(-a, a);


    for (int i = 0; i < n_hidden; ++i)
        hbias[i] = 0.0;

    for (int i = 0; i < rating; ++i)
        for (int j = 0; j < n_visible; ++j)
            vbias[i][j] = 0.0;
}

void RBM::reconstruct(int test_data[][1682], double reconstruct_data[][1682])
{
    double h[num_of_hidden];
    double temp = 0;

    for (int i = 0; i < n_hidden; ++i)
    {
        h[i] = Vtoh_sigm(test_data, W[i], hbias[i]);
    }

    for (int i = 0; i < rating; ++i)
    {
        for (int j = 0; j < n_visible; ++j)
        {
            temp = 0;
            for (int kk = 0; kk < n_hidden; ++kk)
            {
                temp += W[kk][j][i] * h[kk];
            }
            temp += vbias[i][j];
            reconstruct_data[i][j] = sigmoid(temp);
        }
    }
}

double uniform(double min, double max)
{
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

int binomial(double p)
{
    if (p < 0 || p > 1) return 0;
    double r = rand() / (RAND_MAX + 1.0);
    if (r < p) return 1;
    else return 0;
}

double make_predict(RBM rbm, int train_data[][1682], int u, vector<pair<int, int> >& v)
{
    double hidden[num_of_hidden];
    for (int i = 0; i < num_of_hidden; ++i)
    {
        double temp = 0.0;
        for (int j = 0; j < num_of_rating; ++j)
        {
            for (int kk = 0; kk < num_of_movies; ++kk)
            {
                temp += train_data[j][kk] * rbm.W[i][kk][j];
            }
        }
        temp += rbm.hbias[i];
        hidden[i] = rbm.sigmoid(temp);
    }
    int size = v.size();
    double ret = 0;
    for (int i = 0; i < size; ++i)
    {
        double vp[num_of_rating];
        int item = v[i].first;
        int real_rating = v[i].second;

        for (int j = 0; j < num_of_rating; ++j)
        {
            double temp = 0;
            for (int kk = 0; kk < num_of_hidden; ++kk)
            {
                temp += hidden[kk]*rbm.W[kk][item][j];
            }
            temp += rbm.vbias[j][item];
            temp = exp(temp);
            vp[j] = temp;
        }
        double mx = 0, mxi = 0;
        for (int j = 0; j < num_of_rating; ++j)
        {
            if (vp[j] > mx) mx = vp[j], mxi = j;
        }
        ret += (mxi - real_rating) * (mxi - real_rating);
    }
    return ret;
}

void get_train_data(int train_data[][5][1682])
{
    FILE *fp;
    freopen("E:\\DL\\MovieLens\\ml-100k\\u1.base", "r", stdin);
    int u, m, r;
    long long t;
    printf("a\n");
    long long int cnt = 0;
    while (~scanf("%d %d %d %lld", &u, &m, &r, &t))
    {
        u--, m--, r--;
        train_data[u][r][m] = 1;
    }
    fclose(stdin);
}

void get_test_data(vector<pair<int, int> > td[])
{
    FILE* fp;
    freopen("E:\\DL\\MovieLens\\ml-100k\\u1.test", "r", stdin);
    int u, m, r;
    long long t;
    while (~scanf("%d %d %d %lld", &u, &m, &r, &t))
    {
        u--, m--, r--;
        td[u].push_back(make_pair(m, r));
    }
    fclose(stdin);
}

void train()
{
    srand(0);
    int train_N = 100;
    int n_visible = num_of_visible;
    int n_hidden = num_of_hidden;
    int rating = num_of_rating;
    int train_iter = 1000;
    double learning_rate = 0.0001;
    int training_num = 1000;
    int k = 1;
    int train_data[943][5][1682];
    memset(train_data, 0, sizeof(train_data));
    get_train_data(train_data);

    double hbias[num_of_user][num_of_hidden];
    memset(hbias, 0, sizeof(hbias));

    vector<pair<int, int> > test_data[num_of_user];
    get_test_data(test_data);


    RBM rbm = RBM(train_N, n_visible, n_hidden, rating);

    for (int iter = 0; iter < train_iter; ++iter)
    {
        for (int i = 0; i < num_of_user; ++i)
        {
            rbm.contrastiveDivergence(train_data[i], learning_rate, 1);
        }
        int cnt = 0;
        double error = 0;
        for (int i = 0; i < num_of_user; ++i)
        {
            error += make_predict(rbm, train_data[i], i, test_data[i]);
            cnt += test_data[i].size();
        }
        double rmse = sqrt(error / cnt);
        printf("epoch: %d, rmse: %f\n",iter, rmse);
        learning_rate *= 0.9;
    }

    for (int i = 0; i < num_of_hidden; ++i)
        printf("%lf ", rbm.hbias[i]);
    printf("-----------------------------");

    int cnt = 0;
    double error = 0;
    for (int i = 0; i < num_of_user; ++i)
    {
        error += make_predict(rbm, train_data[i], i, test_data[i]);
        cnt += test_data[i].size();
    }
    double rmse = sqrt(error / cnt);
    printf("rmse: %f\n", rmse);

}

// 943 users
// 1682 items
// 100000 ratings

int main()
{
    train();

    return 0;
}
