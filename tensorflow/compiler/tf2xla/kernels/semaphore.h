#ifndef SEMAPHORE_HPP_INCLUDED
#define SEMAPHORE_HPP_INCLUDED

#include <mutex>
#include <condition_variable>

class Semaphore
{
private:
    std::mutex mutex_;
    std::condition_variable condition_;
    unsigned long count_ = 0;

public:
    void notify() {
        //std::cout << "notify" << std::endl;
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        ++count_;
        condition_.notify_one();
    }

    void notify_multiple(int n) {
      //std::cout << "notify_multiple" << std::endl;
      std::lock_guard<decltype(mutex_)> lock(mutex_);
      count_ += n;
      //std::cout << "notify_all in notify_multiple" << std::endl;
      condition_.notify_all();
    }

    void wait() {
        std::unique_lock<decltype(mutex_)> lock(mutex_);
        while(!count_) {
            condition_.wait(lock);
            //std::cout << "woke. Count: " << count_ << std::endl;
        }
        --count_;
    }

    bool try_wait() {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        if(count_) {
            --count_;
            return true;
        }
        return false;
    }
};

#endif

