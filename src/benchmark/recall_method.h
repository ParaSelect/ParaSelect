#pragma once

typedef enum IVFRecallMethod {
    IVF_RECALL_IDX_INTERSECTION = 0,
    IVF_RECALL_THRESHOLD = 1,
    IVF_RECALL_EXACT_RECOMPUTE = 2,
    IVF_RECALL_ADC_THRESHOLD = 3
} IVFRecallMethod;

inline const char* recall_method_name(int method) {
    switch (method) {
        case IVF_RECALL_IDX_INTERSECTION: return "idx_intersection";
        case IVF_RECALL_THRESHOLD: return "threshold";
        case IVF_RECALL_EXACT_RECOMPUTE: return "exact_recompute";
        case IVF_RECALL_ADC_THRESHOLD: return "adc_threshold";
        default: return "idx_intersection";
    }
}

inline int normalize_recall_method(int method) {
    switch (method) {
        case IVF_RECALL_IDX_INTERSECTION:
        case IVF_RECALL_THRESHOLD:
        case IVF_RECALL_EXACT_RECOMPUTE:
        case IVF_RECALL_ADC_THRESHOLD:
            return method;
        default:
            return IVF_RECALL_IDX_INTERSECTION;
    }
}
