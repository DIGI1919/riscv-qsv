// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary model header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef VERILATED_VTILE_H_
#define VERILATED_VTILE_H_  // guard

#include "verilated.h"
#include "verilated_threads.h"

class VTile__Syms;
class VTile___024root;

// This class is the main interface to the Verilated model
class VTile VL_NOT_FINAL : public VerilatedModel {
  private:
    // Symbol table holding complete model state (owned by this class)
    VTile__Syms* const vlSymsp;

  public:

    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_OUT(&io_nasti_aw_bits_addr,31,0);
    VL_IN8(&io_nasti_aw_ready,0,0);
    VL_IN8(&io_nasti_w_ready,0,0);
    VL_OUT8(&io_nasti_w_valid,0,0);
    VL_OUT64(&io_nasti_w_bits_data,63,0);
    VL_IN8(&io_nasti_r_bits_last,0,0);
    VL_IN8(&io_host_fromhost_valid,0,0);
    VL_IN(&io_host_fromhost_bits,31,0);
    VL_OUT8(&io_nasti_b_ready,0,0);
    VL_OUT8(&io_nasti_r_ready,0,0);
    VL_IN64(&io_nasti_r_bits_data,63,0);
    VL_IN8(&io_nasti_b_valid,0,0);
    VL_OUT8(&io_nasti_w_bits_last,0,0);
    VL_IN8(&reset,0,0);
    VL_IN8(&io_nasti_r_valid,0,0);
    VL_OUT8(&io_nasti_ar_valid,0,0);
    VL_OUT(&io_nasti_ar_bits_addr,31,0);
    VL_OUT(&io_host_tohost,31,0);
    VL_IN8(&io_nasti_ar_ready,0,0);
    VL_OUT8(&io_nasti_aw_valid,0,0);
    VL_IN8(&clock,0,0);
    VL_IN8(&io_nasti_b_bits_id,4,0);
    VL_IN8(&io_nasti_b_bits_resp,1,0);
    VL_IN8(&io_nasti_r_bits_id,4,0);
    VL_IN8(&io_nasti_r_bits_resp,1,0);
    VL_OUT8(&io_nasti_aw_bits_id,4,0);
    VL_OUT8(&io_nasti_aw_bits_len,7,0);
    VL_OUT8(&io_nasti_aw_bits_size,2,0);
    VL_OUT8(&io_nasti_aw_bits_burst,1,0);
    VL_OUT8(&io_nasti_aw_bits_lock,0,0);
    VL_OUT8(&io_nasti_aw_bits_cache,3,0);
    VL_OUT8(&io_nasti_aw_bits_prot,2,0);
    VL_OUT8(&io_nasti_aw_bits_qos,3,0);
    VL_OUT8(&io_nasti_w_bits_strb,7,0);
    VL_OUT8(&io_nasti_ar_bits_id,4,0);
    VL_OUT8(&io_nasti_ar_bits_len,7,0);
    VL_OUT8(&io_nasti_ar_bits_size,2,0);
    VL_OUT8(&io_nasti_ar_bits_burst,1,0);
    VL_OUT8(&io_nasti_ar_bits_lock,0,0);
    VL_OUT8(&io_nasti_ar_bits_cache,3,0);
    VL_OUT8(&io_nasti_ar_bits_prot,2,0);
    VL_OUT8(&io_nasti_ar_bits_qos,3,0);

    // CELLS
    // Public to allow access to /* verilator public */ items.
    // Otherwise the application code can consider these internals.

    // Root instance pointer to allow access to model internals,
    // including inlined /* verilator public_flat_* */ items.
    VTile___024root* const rootp;

    // CONSTRUCTORS
    /// Construct the model; called by application code
    /// If contextp is null, then the model will use the default global context
    /// If name is "", then makes a wrapper with a
    /// single model invisible with respect to DPI scope names.
    explicit VTile(VerilatedContext* contextp, const char* name = "TOP");
    explicit VTile(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    virtual ~VTile();
  private:
    VL_UNCOPYABLE(VTile);  ///< Copying not allowed

  public:
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    /// Are there scheduled events to handle?
    bool eventsPending();
    /// Returns time at next time slot. Aborts if !eventsPending()
    uint64_t nextTimeSlot();
    /// Retrieve name of this model instance (as passed to constructor).
    const char* name() const;

    // Abstract methods from VerilatedModel
    const char* hierName() const override final;
    const char* modelName() const override final;
    unsigned threads() const override final;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
