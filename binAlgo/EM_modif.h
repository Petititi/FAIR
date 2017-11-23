
#pragma warning(disable:4251)
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>


class EM_modif
{
public:
    // Type of covariation cv::Matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

     EM_modif(int nclusters, int covMatType, double initVar1, double initVar2,
       const cv::TermCriteria& termCrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                 EM_modif::DEFAULT_MAX_ITERS, FLT_EPSILON));

     EM_modif(int nclusters=EM_modif::DEFAULT_NCLUSTERS, int covMatType=EM_modif::COV_MAT_DEFAULT,
       const cv::TermCriteria& termCrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                 EM_modif::DEFAULT_MAX_ITERS, FLT_EPSILON));

    virtual ~EM_modif();
     virtual void clear();

     virtual bool train(cv::InputArray samples,
                       cv::OutputArray logLikelihoods=cv::noArray(),
                       cv::OutputArray labels=cv::noArray(),
                       cv::OutputArray probs=cv::noArray());

     virtual bool trainE(cv::InputArray samples,
                        cv::InputArray means0,
                        cv::InputArray covs0=cv::noArray(),
                        cv::InputArray weights0=cv::noArray(),
                        cv::OutputArray logLikelihoods=cv::noArray(),
                        cv::OutputArray labels=cv::noArray(),
                        cv::OutputArray probs=cv::noArray());

     virtual bool trainM(cv::InputArray samples,
                        cv::InputArray probs0,
                        cv::OutputArray logLikelihoods=cv::noArray(),
                        cv::OutputArray labels=cv::noArray(),
                        cv::OutputArray probs=cv::noArray());

     cv::Vec2d predict(cv::InputArray sample,
                cv::OutputArray probs=cv::noArray()) const;

     bool isTrained() const;


protected:

    virtual void setTrainData(int startStep, const cv::Mat& samples,
                              const cv::Mat* probs0,
                              const cv::Mat* means0,
                              const std::vector<cv::Mat>* covs0,
                              const cv::Mat* weights0);

    bool doTrain(int startStep,
                 cv::OutputArray logLikelihoods,
                 cv::OutputArray labels,
                 cv::OutputArray probs);
    virtual void eStep();
    virtual void mStep();

    void clusterTrainSamples();
    void decomposeCovs();
    void computeLogWeightDivDet();

    cv::Vec2d computeProbabilities(const cv::Mat& sample, cv::Mat* probs) const;

public:
    // all inner cv::Matrices have type CV_64FC1
     int nclusters;
     int covMatType;
     int maxIters;
     double epsilon;

    cv::Mat trainSamples;
    cv::Mat trainProbs;
    cv::Mat trainLogLikelihoods;
    cv::Mat trainLabels;

    cv::Mat weights;
    cv::Mat means;
    std::vector<cv::Mat> covs;

    std::vector<cv::Mat> covsEigenValues;
    std::vector<cv::Mat> covsRotateMats;
    std::vector<cv::Mat> invCovsEigenValues;
    cv::Mat logWeightDivDet;
    double initVar[2];
};