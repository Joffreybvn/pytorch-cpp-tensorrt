
#ifndef TENSORRTMODEL_TRT_DESTROY_H
#define TENSORRTMODEL_TRT_DESTROY_H


/**
 * Destroy TensorRT objects if something goes wrong.
 */
struct TRTDestroy {

    template <class T>
    void operator()(T* obj) const {

        if (obj) {
            obj->destroy();
        }
    }
};


#endif
