#ifndef LIGHTGBM_BOOSTING_BOOSTING_FOREST_H_
#define LIGHTGBM_BOOSTING_BOOSTING_FOREST_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

//#include <iostream>

namespace LightGBM {
/*!
* \brief Boosting Forest algorithm implementation. including Training, prediction, bagging.
*/
class BF: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  BF() : GBDT() { }
  /*!
  * \brief Destructor
  */
  ~BF() { }
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    random_for_drop_ = Random(gbdt_config_->drop_seed);
    sum_weight_ = 0.0f;
  }

  void ResetConfig(const BoostingConfig* config) override {
    GBDT::ResetConfig(config);
    random_for_drop_ = Random(gbdt_config_->drop_seed);
    sum_weight_ = 0.0f;
  }

  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian) override {
    is_update_score_cur_iter_ = false;
    bool ret = GBDT::TrainOneIter(gradient, hessian);
    if (ret) {
      return ret;
    }
    // normalize
     Normalize();
    if (!gbdt_config_->uniform_drop) {
      Log::Fatal("Boosting Forest doest not support non-uniform drop temporarily");
//      tree_weight_.push_back(shrinkage_rate_);
//      sum_weight_ += shrinkage_rate_;
    }
    return false;
  }

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  const double* GetTrainingScore(int64_t* out_len) override {
    if (!is_update_score_cur_iter_) {
      // only drop one time in one iteration
      DroppingTrees();
      is_update_score_cur_iter_ = true;
    }
    *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_class_;
    return train_score_updater_->score();
  }

private:
  /*!
  * \brief drop trees based on drop_rate
  */
  void DroppingTrees() {

    /*!
     * Xixuan: drop trees by sections
     */

    if (gbdt_config_->bagging_fraction == 1.0f) {
      Log::Warning("Boosting Forest should use 0.0 < bagging_fraction < 1.0");
    }

    drop_index_.clear();

    int num_tree_per_forest = gbdt_config_->num_tree_per_forest;
    int num_tree_to_drop = num_tree_per_forest * gbdt_config_->drop_rate;

    // select dropping tree indices based on drop_rate
    if (!gbdt_config_->uniform_drop) {
      Log::Fatal("Boosting Forest doest not support non-uniform drop temporarily");
    } else {
      if (iter_ > 0) {
        int current_forest_idx = (num_init_iteration_ + iter_ - 1) / num_tree_per_forest;

//          std::cout << num_tree_to_drop << " " << current_forest_idx << std::endl;

        if (num_tree_to_drop < 1) {
          Log::Warning("Boosting Forest a has too small drop_rate and thus does not drop trees for each forest");
        } else {
          // drop trees for each forest
          for (int forest_idx = 0; forest_idx < current_forest_idx; ++forest_idx) {
            std::vector<int> dropped_tree_idx_in_forest = random_for_drop_.Sample(
                    num_tree_per_forest, num_tree_to_drop
            );

//            for (int idx : dropped_tree_idx_in_forest) {
//              std::cout << idx << " ";
//            }
//            std::cout << std::endl;

            for (int tree_idx : dropped_tree_idx_in_forest) {
              drop_index_.push_back(num_init_iteration_ + forest_idx * num_tree_per_forest + tree_idx);
            }
          }
          // drop the trees in the current forest
          for (int tree_idx = current_forest_idx * num_tree_per_forest; tree_idx < iter_; ++ tree_idx) {
            drop_index_.push_back(tree_idx);
          }
        }
      }
    }

//    std::cout << "indice for dropping" << std::endl;
//    for (int idx : drop_index_) {
//      std::cout << idx << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "before dropping" << std::endl;
//    for (int idx = 0; idx < 3; ++idx) {
//      std::cout << *(train_score_updater_->score() + idx) << " ";
//    }
//    std::cout << std::endl;

    // drop trees
    for (auto i : drop_index_) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
        models_[curr_tree]->Shrinkage(-1.0);
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
        if (curr_tree == 0) {
          train_score_updater_->AddScore(models_[curr_tree]->bias(), cur_tree_id);
        }
      }
    }

//    std::cout << "after dropping" << std::endl;
//    for (int idx = 0; idx < 3; ++idx) {
//      std::cout << *(train_score_updater_->score() + idx) << " ";
//    }
//    std::cout << std::endl;

    if (!gbdt_config_->xgboost_dart_mode) {
      shrinkage_rate_ = gbdt_config_->learning_rate / (1.0f + static_cast<double>(drop_index_.size()));
    } else {
      Log::Fatal("Boosting Forest does not support xgboost_dart_mode");
//      if (drop_index_.empty()) {
//        shrinkage_rate_ = gbdt_config_->learning_rate;
//      } else {
//        shrinkage_rate_ = gbdt_config_->learning_rate / (gbdt_config_->learning_rate + static_cast<double>(drop_index_.size()));
//      }
    }
  }
  /*!
  * \brief normalize dropped trees
  * NOTE: num_drop_tree(k), learning_rate(lr), shrinkage_rate_ = lr / (k + 1)
  *       step 1: shrink tree to -1 -> drop tree
  *       step 2: shrink tree to k / (k + 1) - 1 from -1, by 1/(k+1)
  *               -> normalize for valid data
  *       step 3: shrink tree to k / (k + 1) from k / (k + 1) - 1, by -k
  *               -> normalize for train data
  *       end with tree weight = (k / (k + 1)) * old_weight
  */
  void Normalize() {
//    double k = static_cast<double>(drop_index_.size());
    if (!gbdt_config_->xgboost_dart_mode) {
      for (auto i : drop_index_) {
        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
          auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
          // update validation score
//          models_[curr_tree]->Shrinkage(1.0f / (k + 1.0f));
//          for (auto& score_updater : valid_score_updater_) {
//            score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
//          }
          // update training score
          models_[curr_tree]->Shrinkage(-1.0);
          train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
          if (curr_tree == 0) {
            train_score_updater_->AddScore(-models_[curr_tree]->bias(), cur_tree_id);
          }
        }
        if (!gbdt_config_->uniform_drop) {
          Log::Fatal("Boosting Forest doest not support non-uniform drop temporarily");
        }
      }
    } else {
      Log::Fatal("Boosting Forest does not support xgboost_dart_mode");
//      for (auto i : drop_index_) {
//        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
//          auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
//          // update validation score
//          models_[curr_tree]->Shrinkage(shrinkage_rate_);
//          for (auto& score_updater : valid_score_updater_) {
//            score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
//          }
//          // update training score
//          models_[curr_tree]->Shrinkage(-k / gbdt_config_->learning_rate);
//          train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
//        }
//        if (!gbdt_config_->uniform_drop) {
//          Log::Fatal("Boosting Forest doest not support non-uniform drop temporarily");
//        }
//      }
    }
  }
  /*! \brief The weights of all trees, used to choose drop trees */
  std::vector<double> tree_weight_;
  /*! \brief sum weights of all trees */
  double sum_weight_;
  /*! \brief The indices of dropping trees */
  std::vector<int> drop_index_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
  /*! \brief Flag that the score is update on current iter or not*/
  bool is_update_score_cur_iter_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_BOOSTING_FOREST_H_
