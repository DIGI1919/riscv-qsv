// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VTile.h for the primary calling header

#include "verilated.h"

#include "VTile___024root.h"

VL_ATTR_COLD void VTile___024root___eval_static(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_static\n"); );
}

VL_ATTR_COLD void VTile___024root___eval_initial__TOP(VTile___024root* vlSelf);

VL_ATTR_COLD void VTile___024root___eval_initial(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_initial\n"); );
    // Body
    VTile___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vtrigrprev__TOP__clock = vlSelf->clock;
}

VL_ATTR_COLD void VTile___024root___eval_initial__TOP(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_initial__TOP\n"); );
    // Body
    vlSelf->io_nasti_aw_bits_id = 0U;
    vlSelf->io_nasti_aw_bits_len = 1U;
    vlSelf->io_nasti_aw_bits_size = 3U;
    vlSelf->io_nasti_aw_bits_burst = 1U;
    vlSelf->io_nasti_aw_bits_lock = 0U;
    vlSelf->io_nasti_aw_bits_cache = 0U;
    vlSelf->io_nasti_aw_bits_prot = 0U;
    vlSelf->io_nasti_aw_bits_qos = 0U;
    vlSelf->io_nasti_w_bits_strb = 0xffU;
    vlSelf->io_nasti_ar_bits_id = 0U;
    vlSelf->io_nasti_ar_bits_len = 1U;
    vlSelf->io_nasti_ar_bits_size = 3U;
    vlSelf->io_nasti_ar_bits_burst = 1U;
    vlSelf->io_nasti_ar_bits_lock = 0U;
    vlSelf->io_nasti_ar_bits_cache = 0U;
    vlSelf->io_nasti_ar_bits_prot = 0U;
    vlSelf->io_nasti_ar_bits_qos = 0U;
}

VL_ATTR_COLD void VTile___024root___eval_final(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_final\n"); );
}

VL_ATTR_COLD void VTile___024root___eval_triggers__stl(VTile___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__stl(VTile___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD void VTile___024root___eval_stl(VTile___024root* vlSelf);

VL_ATTR_COLD void VTile___024root___eval_settle(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_settle\n"); );
    // Init
    CData/*0:0*/ __VstlContinue;
    // Body
    vlSelf->__VstlIterCount = 0U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        __VstlContinue = 0U;
        VTile___024root___eval_triggers__stl(vlSelf);
        if (vlSelf->__VstlTriggered.any()) {
            __VstlContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VstlIterCount))) {
#ifdef VL_DEBUG
                VTile___024root___dump_triggers__stl(vlSelf);
#endif
                VL_FATAL_MT("/home/digi/code/c/riscv-qv/Tile.sv", 3491, "", "Settle region did not converge.");
            }
            vlSelf->__VstlIterCount = ((IData)(1U) 
                                       + vlSelf->__VstlIterCount);
            VTile___024root___eval_stl(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__stl(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if (vlSelf->__VstlTriggered.at(0U)) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void VTile___024root___stl_sequent__TOP__0(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___stl_sequent__TOP__0\n"); );
    // Init
    VlWide<4>/*127:0*/ Tile__DOT__dcache__DOT___GEN_3;
    VL_ZERO_W(128, Tile__DOT__dcache__DOT___GEN_3);
    CData/*0:0*/ Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0;
    Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0 = 0;
    VlWide<8>/*255:0*/ __Vtemp_hd0742a16__0;
    VlWide<8>/*255:0*/ __Vtemp_h5336a7a2__0;
    VlWide<8>/*255:0*/ __Vtemp_h0dd6d4d8__0;
    VlWide<8>/*255:0*/ __Vtemp_hc380ab92__0;
    // Body
    vlSelf->io_host_tohost = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_1 
        = (0x1ffffffffULL & (4ULL + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc));
    vlSelf->Tile__DOT___arb_io_dcache_aw_ready = ((~ (IData)(
                                                             (0U 
                                                              != (IData)(vlSelf->Tile__DOT__arb__DOT__state)))) 
                                                  & (IData)(vlSelf->io_nasti_aw_ready));
    vlSelf->io_nasti_r_ready = (((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                 & (1U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))) 
                                | ((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                   & (2U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    if (((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0))) {
        vlSelf->Tile__DOT__icache__DOT__rdata[0U] = 
            vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory
            [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN];
        vlSelf->Tile__DOT__icache__DOT__rdata[1U] = 
            vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory
            [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN];
    } else {
        vlSelf->Tile__DOT__icache__DOT__rdata[0U] = 0U;
        vlSelf->Tile__DOT__icache__DOT__rdata[1U] = 0U;
    }
    vlSelf->Tile__DOT__icache__DOT__rdata[2U] = (IData)(
                                                        (((QData)((IData)(
                                                                          (((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                            & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                            ? 
                                                                           vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                                                                           [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN]
                                                                            : 0U))) 
                                                          << 0x20U) 
                                                         | (QData)((IData)(
                                                                           (((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                             & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                             ? 
                                                                            vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                                                                            [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN]
                                                                             : 0U)))));
    vlSelf->Tile__DOT__icache__DOT__rdata[3U] = (IData)(
                                                        ((((QData)((IData)(
                                                                           (((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                             & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                             ? 
                                                                            vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                                                                            [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN]
                                                                             : 0U))) 
                                                           << 0x20U) 
                                                          | (QData)((IData)(
                                                                            (((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                              & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                              ? 
                                                                             vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                                                                             [vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN]
                                                                              : 0U)))) 
                                                         >> 0x20U));
    vlSelf->Tile__DOT___arb_io_dcache_w_ready = ((IData)(vlSelf->io_nasti_w_ready) 
                                                 & (3U 
                                                    == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
    vlSelf->Tile__DOT__icache__DOT___wmask_T_2 = (0x7ffffU 
                                                  & ((IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask) 
                                                     << 
                                                     (0xcU 
                                                      & vlSelf->Tile__DOT__icache__DOT__addr_reg)));
    vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 = (0x7ffffU 
                                                  & ((IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask) 
                                                     << 
                                                     (0xcU 
                                                      & vlSelf->Tile__DOT__dcache__DOT__addr_reg)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out 
        = ((0xc00U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                       >> 0x14U)) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle
            : ((0xc01U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                           >> 0x14U)) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0
                : ((0xc02U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                               >> 0x14U)) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret
                    : ((0xc80U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                   >> 0x14U)) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh
                        : ((0xc81U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                       >> 0x14U)) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh
                            : ((0xc82U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                           >> 0x14U))
                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth
                                : ((0x900U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                               >> 0x14U))
                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle
                                    : ((0x901U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                   >> 0x14U))
                                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0
                                        : ((0x902U 
                                            == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                >> 0x14U))
                                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret
                                            : ((0x980U 
                                                == 
                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                 >> 0x14U))
                                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh
                                                : (
                                                   (0x981U 
                                                    == 
                                                    (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                     >> 0x14U))
                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh
                                                    : 
                                                   ((0x982U 
                                                     == 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                      >> 0x14U))
                                                     ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth
                                                     : 
                                                    ((0xf00U 
                                                      == 
                                                      (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                       >> 0x14U))
                                                      ? 0x100100U
                                                      : 
                                                     (((0xf01U 
                                                        == 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                         >> 0x14U)) 
                                                       | (0xf10U 
                                                          == 
                                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                           >> 0x14U)))
                                                       ? 0U
                                                       : 
                                                      ((0x301U 
                                                        == 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                         >> 0x14U))
                                                        ? 0x100U
                                                        : 
                                                       ((0x302U 
                                                         == 
                                                         (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                          >> 0x14U))
                                                         ? 0U
                                                         : 
                                                        ((0x304U 
                                                          == 
                                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                           >> 0x14U))
                                                          ? 
                                                         (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIE) 
                                                           << 7U) 
                                                          | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIE) 
                                                             << 3U))
                                                          : 
                                                         ((0x321U 
                                                           == 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                            >> 0x14U))
                                                           ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtimecmp
                                                           : 
                                                          ((0x701U 
                                                            == 
                                                            (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                             >> 0x14U))
                                                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0
                                                            : 
                                                           ((0x741U 
                                                             == 
                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                              >> 0x14U))
                                                             ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh
                                                             : 
                                                            ((0x340U 
                                                              == 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                               >> 0x14U))
                                                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mscratch
                                                              : 
                                                             ((0x341U 
                                                               == 
                                                               (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                >> 0x14U))
                                                               ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc
                                                               : 
                                                              ((0x342U 
                                                                == 
                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                 >> 0x14U))
                                                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mcause
                                                                : 
                                                               ((0x343U 
                                                                 == 
                                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                  >> 0x14U))
                                                                 ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mbadaddr
                                                                 : 
                                                                ((0x344U 
                                                                  == 
                                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                   >> 0x14U))
                                                                  ? 
                                                                 (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIP) 
                                                                   << 7U) 
                                                                  | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIP) 
                                                                     << 3U))
                                                                  : 
                                                                 ((0x780U 
                                                                   == 
                                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                    >> 0x14U))
                                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost
                                                                   : 
                                                                  ((0x781U 
                                                                    == 
                                                                    (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                     >> 0x14U))
                                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mfromhost
                                                                    : 
                                                                   ((0x300U 
                                                                     == 
                                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                      >> 0x14U))
                                                                     ? 
                                                                    (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1) 
                                                                      << 4U) 
                                                                     | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1) 
                                                                         << 3U) 
                                                                        | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV) 
                                                                            << 1U) 
                                                                           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE))))
                                                                     : 
                                                                    ((0x721U 
                                                                      == 
                                                                      (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                       >> 0x14U))
                                                                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm
                                                                      : 
                                                                     ((0x722U 
                                                                       == 
                                                                       (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                        >> 0x14U))
                                                                       ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qstate
                                                                       : 0U))))))))))))))))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17 
        = ((0x7fffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                             >> 0x1dU)))) 
                           << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                               >> 0x16U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19 
        = (((QData)((IData)((0x7fffffffU & (- (IData)(
                                                      (1U 
                                                       & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                          >> 0x1dU))))))) 
            << 8U) | (QData)((IData)((0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                               >> 0x16U)))));
    if (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0))) {
        vlSelf->Tile__DOT__dcache__DOT__rdata[0U] = 
            vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory
            [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN];
        vlSelf->Tile__DOT__dcache__DOT__rdata[1U] = 
            vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory
            [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN];
    } else {
        vlSelf->Tile__DOT__dcache__DOT__rdata[0U] = 0U;
        vlSelf->Tile__DOT__dcache__DOT__rdata[1U] = 0U;
    }
    vlSelf->Tile__DOT__dcache__DOT__rdata[2U] = (IData)(
                                                        (((QData)((IData)(
                                                                          (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                            & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                            ? 
                                                                           vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                                                                           [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN]
                                                                            : 0U))) 
                                                          << 0x20U) 
                                                         | (QData)((IData)(
                                                                           (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                             & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                             ? 
                                                                            vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                                                                            [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN]
                                                                             : 0U)))));
    vlSelf->Tile__DOT__dcache__DOT__rdata[3U] = (IData)(
                                                        ((((QData)((IData)(
                                                                           (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                             & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                             ? 
                                                                            vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                                                                            [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN]
                                                                             : 0U))) 
                                                           << 0x20U) 
                                                          | (QData)((IData)(
                                                                            (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                                              & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0))
                                                                              ? 
                                                                             vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                                                                             [vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN]
                                                                              : 0U)))) 
                                                         >> 0x20U));
    vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0 
        = (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                 | (1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))));
    vlSelf->Tile__DOT__icache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (1U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT__dcache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (2U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           & (IData)((0U == (0x10100000U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           & (IData)((0x100000U == (0x10100000U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd) 
               >> 1U) & (0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                         >> 0xfU)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc_check) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
              >> 1U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid 
        = (1U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)) 
                  | (2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)))
                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu
                  : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)) 
                     & (0U != (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid 
        = (1U & ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type))
                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu
                  : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type)) 
                     & (0U != (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT___io_expt_T_12 
        = ((3U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                  >> 0x1cU)) > (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV));
    VL_SHIFTR_WWI(256,256,8, __Vtemp_hd0742a16__0, vlSelf->Tile__DOT__icache__DOT__v, 
                  (0xffU & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0 
        = (1U & __Vtemp_hd0742a16__0[0U]);
    vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
        = (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
            & (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_0))
            ? vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory
           [vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN]
            : 0U);
    VL_SHIFTR_WWI(256,256,8, __Vtemp_h5336a7a2__0, vlSelf->Tile__DOT__dcache__DOT__v, 
                  (0xffU & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                            >> 4U)));
    Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0 
        = (1U & __Vtemp_h5336a7a2__0[0U]);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19 
        = ((0x3ffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x1dU)))) 
                          << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                              >> 0x16U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21 
        = ((0x3ffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x10U)))) 
                          << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                              >> 9U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0 
        = ((0x3ff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                          >> 0x1dU)))) 
                        << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x16U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2 
        = ((0x3ff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                          >> 0x10U)))) 
                        << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 9U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1 
        = ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                            >> 0xfU))) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory
           [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                      >> 0xfU))] : 0U);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2 
        = ((0x73U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
           | (0x100073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out 
        = ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul
            : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))
                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe
                : vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__bypass__DOT____VdfgTmp_h8e44abc1__0 
        = ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)) 
           | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)) 
              | (5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10 
        = ((0x37U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
           | (0x17U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75 
        = (0xfU == ((0xf00000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                  >> 8U)) | (0xfffffU 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN = 
        ((0x380U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                    >> 5U)) | (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
    if (vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg) {
        Tile__DOT__dcache__DOT___GEN_3[0U] = (IData)(vlSelf->Tile__DOT__dcache__DOT__refill_buf_0);
        Tile__DOT__dcache__DOT___GEN_3[1U] = (IData)(
                                                     (vlSelf->Tile__DOT__dcache__DOT__refill_buf_0 
                                                      >> 0x20U));
        Tile__DOT__dcache__DOT___GEN_3[2U] = (IData)(vlSelf->Tile__DOT__dcache__DOT__refill_buf_1);
        Tile__DOT__dcache__DOT___GEN_3[3U] = (IData)(
                                                     (vlSelf->Tile__DOT__dcache__DOT__refill_buf_1 
                                                      >> 0x20U));
    } else if (vlSelf->Tile__DOT__dcache__DOT__ren_reg) {
        Tile__DOT__dcache__DOT___GEN_3[0U] = vlSelf->Tile__DOT__dcache__DOT__rdata[0U];
        Tile__DOT__dcache__DOT___GEN_3[1U] = vlSelf->Tile__DOT__dcache__DOT__rdata[1U];
        Tile__DOT__dcache__DOT___GEN_3[2U] = vlSelf->Tile__DOT__dcache__DOT__rdata[2U];
        Tile__DOT__dcache__DOT___GEN_3[3U] = vlSelf->Tile__DOT__dcache__DOT__rdata[3U];
    } else {
        Tile__DOT__dcache__DOT___GEN_3[0U] = vlSelf->Tile__DOT__dcache__DOT__rdata_buf[0U];
        Tile__DOT__dcache__DOT___GEN_3[1U] = vlSelf->Tile__DOT__dcache__DOT__rdata_buf[1U];
        Tile__DOT__dcache__DOT___GEN_3[2U] = vlSelf->Tile__DOT__dcache__DOT__rdata_buf[2U];
        Tile__DOT__dcache__DOT___GEN_3[3U] = vlSelf->Tile__DOT__dcache__DOT__rdata_buf[3U];
    }
    vlSelf->Tile__DOT__dcache__DOT___GEN_4 = ((IData)(vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0) 
                                              | (2U 
                                                 == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
    vlSelf->Tile__DOT__icache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__icache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__read_count));
    vlSelf->Tile__DOT__dcache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_count));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__illegal) 
           | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid) 
              | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid) 
                 | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid) 
                    | (((0U != (3U & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))) 
                        & ((~ ((0xc00U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                           >> 0x14U)) 
                               | ((0xc01U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                              >> 0x14U)) 
                                  | ((0xc02U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                 >> 0x14U)) 
                                     | ((0xc80U == 
                                         (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                          >> 0x14U)) 
                                        | ((0xc81U 
                                            == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                >> 0x14U)) 
                                           | ((0xc82U 
                                               == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                   >> 0x14U)) 
                                              | ((0x900U 
                                                  == 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                   >> 0x14U)) 
                                                 | ((0x901U 
                                                     == 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                      >> 0x14U)) 
                                                    | ((0x902U 
                                                        == 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                         >> 0x14U)) 
                                                       | ((0x980U 
                                                           == 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                            >> 0x14U)) 
                                                          | ((0x981U 
                                                              == 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                               >> 0x14U)) 
                                                             | ((0x982U 
                                                                 == 
                                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                  >> 0x14U)) 
                                                                | ((0xf00U 
                                                                    == 
                                                                    (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                     >> 0x14U)) 
                                                                   | ((0xf01U 
                                                                       == 
                                                                       (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                        >> 0x14U)) 
                                                                      | ((0xf10U 
                                                                          == 
                                                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                           >> 0x14U)) 
                                                                         | ((0x301U 
                                                                             == 
                                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                              >> 0x14U)) 
                                                                            | ((0x302U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                               | ((0x304U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x321U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x701U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x741U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x340U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x341U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x342U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x343U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x344U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x780U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x781U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x300U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | ((0x721U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U)) 
                                                                                | (0x722U 
                                                                                == 
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                                                >> 0x14U))))))))))))))))))))))))))))))))) 
                           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT___io_expt_T_12))) 
                       | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen) 
                           & ((3U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                      >> 0x1eU)) | 
                              ((0x301U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                           >> 0x14U)) 
                               | (0x302U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                             >> 0x14U))))) 
                          | (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
                              & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT___io_expt_T_12)) 
                             | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak)))))))));
    VL_SHIFTR_WWI(256,256,8, __Vtemp_h0dd6d4d8__0, vlSelf->Tile__DOT__icache__DOT__d, 
                  (0xffU & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__icache__DOT__is_dirty = ((IData)(vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0) 
                                                & __Vtemp_h0dd6d4d8__0[0U]);
    vlSelf->Tile__DOT__icache__DOT__hit = ((IData)(vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0) 
                                           & ((((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                & (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_0))
                                                ? vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory
                                               [vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN]
                                                : 0U) 
                                              == (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                  >> 0xcU)));
    vlSelf->io_nasti_aw_bits_addr = ((vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
                                      << 0xcU) | (0xff0U 
                                                  & vlSelf->Tile__DOT__dcache__DOT__addr_reg));
    VL_SHIFTR_WWI(256,256,8, __Vtemp_hc380ab92__0, vlSelf->Tile__DOT__dcache__DOT__d, 
                  (0xffU & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__dcache__DOT__is_dirty = ((IData)(Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0) 
                                                & __Vtemp_hc380ab92__0[0U]);
    vlSelf->Tile__DOT__dcache__DOT__hit = ((IData)(Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0) 
                                           & (vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
                                              == (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                  >> 0xcU)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2) 
           | (0x10000073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__bypass__DOT____VdfgTmp_h8e44abc1__0) 
            & ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                >> 0x14U))) & (((0x1fU 
                                                 & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                    >> 7U)) 
                                                == 
                                                (0x1fU 
                                                 & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                    >> 0x14U))) 
                                               & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en))))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out
            : ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                >> 0x14U))) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory
               [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                          >> 0x14U))] : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__bypass__DOT____VdfgTmp_h8e44abc1__0) 
           & ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                               >> 0xfU))) & (((0x1fU 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                                  >> 7U)) 
                                              == (0x1fU 
                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                     >> 0xfU))) 
                                             & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10) 
           | (0x6fU == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75) 
           | (0x100fU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9 
        = ((0x6fU == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
           | (0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)));
    vlSelf->__VdfgTmp_h0634cb66__0 = ((0x17U == (0x7fU 
                                                 & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
                                      | ((0x6fU == 
                                          (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
                                         | ((0x67U 
                                             == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                            | ((0x63U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                               | ((0xe3U 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                  | ((0x263U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                     | ((0x2e3U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                        | ((0x363U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                           | ((0x3e3U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                              | ((3U 
                                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                 | ((0x83U 
                                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                    | ((0x103U 
                                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                       | ((0x203U 
                                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                          | ((0x283U 
                                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                             | ((0x23U 
                                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                                | ((0xa3U 
                                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                                | ((0x123U 
                                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                                                | (0x13U 
                                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)))))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14 
        = ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1 
        = ((0x7c00U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                       >> 0x11U)) | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3 
        = ((0x2f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x373U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x3f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20 
        = ((0x23U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0xa3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x123U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x83U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x103U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x203U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | (0x283U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8 
        = ((0x63U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0xe3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x263U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x2e3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x363U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       | (0x3e3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0 
        = ((0x1fc00U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                        >> 0xfU)) | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN));
    vlSelf->io_nasti_w_bits_data = ((IData)(vlSelf->Tile__DOT__dcache__DOT__write_count)
                                     ? (((QData)((IData)(
                                                         Tile__DOT__dcache__DOT___GEN_3[3U])) 
                                         << 0x20U) 
                                        | (QData)((IData)(
                                                          Tile__DOT__dcache__DOT___GEN_3[2U])))
                                     : (((QData)((IData)(
                                                         Tile__DOT__dcache__DOT___GEN_3[1U])) 
                                         << 0x20U) 
                                        | (QData)((IData)(
                                                          Tile__DOT__dcache__DOT___GEN_3[0U]))));
    vlSelf->Tile__DOT___dcache_io_cpu_resp_bits_data 
        = Tile__DOT__dcache__DOT___GEN_3[(3U & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                >> 2U))];
    vlSelf->Tile__DOT___dcache_io_nasti_b_ready = (
                                                   (~ 
                                                    ((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_4) 
                                                     | (3U 
                                                        == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                                                   & (4U 
                                                      == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
    vlSelf->Tile__DOT___dcache_io_nasti_w_valid = (
                                                   (~ (IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_4)) 
                                                   & (3U 
                                                      == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
    vlSelf->Tile__DOT__icache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__icache__DOT__read_wrap_out));
    vlSelf->Tile__DOT__dcache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_wrap_out));
    vlSelf->Tile__DOT__icache__DOT___T_35 = ((IData)(vlSelf->Tile__DOT__icache__DOT__hit) 
                                             | (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc_reg));
    vlSelf->Tile__DOT___icache_io_cpu_resp_valid = 
        (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
               | (((1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                   & (IData)(vlSelf->Tile__DOT__icache__DOT__hit)) 
                  | ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask)))) 
                     & (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc_reg)))));
    vlSelf->Tile__DOT__dcache__DOT___T_34 = ((IData)(vlSelf->Tile__DOT__dcache__DOT__hit) 
                                             | (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg));
    vlSelf->Tile__DOT___dcache_io_cpu_resp_valid = 
        (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
               | (((1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__hit)) 
                  | ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask)))) 
                     & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg)))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23) 
           | (0x10200073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
    if (vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out;
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in 
            = ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                >> 0xfU))) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory
               [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                          >> 0xfU))] : vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory
               [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                          >> 0x14U))]);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1;
    }
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_21 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9));
    vlSelf->__VdfgTmp_h8aef8152__0 = (1U & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14)) 
                                            | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h990c992e__0 
        = ((0x57U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
           | ((0x2057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
              | ((0x1057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                 | ((0x3057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                    | ((0xd7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                       | ((0x2d7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                          | ((0x4d7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                             | ((0x457U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                | ((0x557U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                   | ((0x267U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                      | ((0x2e7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                         | ((0x1e7U 
                                             == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                            | ((0xe7U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                               | ((0x167U 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                  | ((0x367U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                     | (0x3e7U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0) 
           | ((0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22 
        = ((0x433U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
           | ((0x4b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
              | ((0x533U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                 | ((0x5b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                    | ((0x633U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                       | ((0x6b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                          | ((0x733U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             | (0x7b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0))))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19 
        = ((0x13U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x113U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x193U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x213U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x313U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       | ((0x393U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                          | ((0x93U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             | ((0x293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                | (0x8293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)))))))));
    vlSelf->__VdfgTmp_hfd3c45f9__0 = ((0x333U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                      | (0x3b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift 
        = (vlSelf->Tile__DOT___dcache_io_cpu_resp_bits_data 
           >> (0x18U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
                        << 3U)));
    vlSelf->io_nasti_b_ready = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_b_ready) 
                                & (4U == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
    vlSelf->io_nasti_w_valid = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid) 
                                & (3U == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
    vlSelf->Tile__DOT__dcache__DOT___T_1 = ((IData)(vlSelf->Tile__DOT___arb_io_dcache_w_ready) 
                                            & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid));
    if (vlSelf->Tile__DOT__icache__DOT__is_alloc) {
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__icache__DOT__wdata[0U] = (IData)(vlSelf->Tile__DOT__icache__DOT__refill_buf_0);
        vlSelf->Tile__DOT__icache__DOT__wdata[1U] = (IData)(
                                                            (vlSelf->Tile__DOT__icache__DOT__refill_buf_0 
                                                             >> 0x20U));
        vlSelf->Tile__DOT__icache__DOT__wdata[2U] = (IData)(vlSelf->io_nasti_r_bits_data);
        vlSelf->Tile__DOT__icache__DOT__wdata[3U] = (IData)(
                                                            (vlSelf->io_nasti_r_bits_data 
                                                             >> 0x20U));
    } else {
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                       >> 4U));
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                       >> 8U));
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                       >> 0xcU));
        vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask 
            = (0xfU & ((IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask) 
                       << (0xcU & vlSelf->Tile__DOT__icache__DOT__addr_reg)));
        vlSelf->Tile__DOT__icache__DOT__wdata[0U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[1U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[2U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[3U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
    }
    if (vlSelf->Tile__DOT__dcache__DOT__is_alloc) {
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask = 0xfU;
        vlSelf->Tile__DOT__dcache__DOT__wdata[0U] = (IData)(vlSelf->Tile__DOT__dcache__DOT__refill_buf_0);
        vlSelf->Tile__DOT__dcache__DOT__wdata[1U] = (IData)(
                                                            (vlSelf->Tile__DOT__dcache__DOT__refill_buf_0 
                                                             >> 0x20U));
        vlSelf->Tile__DOT__dcache__DOT__wdata[2U] = (IData)(vlSelf->io_nasti_r_bits_data);
        vlSelf->Tile__DOT__dcache__DOT__wdata[3U] = (IData)(
                                                            (vlSelf->io_nasti_r_bits_data 
                                                             >> 0x20U));
    } else {
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 
                       >> 4U));
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 
                       >> 8U));
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask 
            = (0xfU & (vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 
                       >> 0xcU));
        vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask 
            = (0xfU & ((IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask) 
                       << (0xcU & vlSelf->Tile__DOT__dcache__DOT__addr_reg)));
        vlSelf->Tile__DOT__dcache__DOT__wdata[0U] = vlSelf->Tile__DOT__dcache__DOT__cpu_data;
        vlSelf->Tile__DOT__dcache__DOT__wdata[1U] = vlSelf->Tile__DOT__dcache__DOT__cpu_data;
        vlSelf->Tile__DOT__dcache__DOT__wdata[2U] = vlSelf->Tile__DOT__dcache__DOT__cpu_data;
        vlSelf->Tile__DOT__dcache__DOT__wdata[3U] = vlSelf->Tile__DOT__dcache__DOT__cpu_data;
    }
    vlSelf->Tile__DOT___icache_io_nasti_ar_valid = 
        ((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
         & ((1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))
             ? ((~ (IData)(vlSelf->Tile__DOT__icache__DOT__hit)) 
                & (~ (IData)(vlSelf->Tile__DOT__icache__DOT__is_dirty)))
             : ((2U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))
                 ? ((~ (IData)(vlSelf->Tile__DOT__icache__DOT___T_35)) 
                    & (~ (IData)(vlSelf->Tile__DOT__icache__DOT__is_dirty)))
                 : ((~ ((3U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                        | (4U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
                    & (5U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))))));
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
            & (IData)(vlSelf->Tile__DOT__icache__DOT___T_35)) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
            & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)) 
               & (IData)(vlSelf->Tile__DOT__dcache__DOT___T_34))) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
    vlSelf->Tile__DOT__dcache__DOT___T_35 = ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_34) 
                                             | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT___icache_io_cpu_resp_valid)) 
                 | (~ (IData)(vlSelf->Tile__DOT___dcache_io_cpu_resp_valid))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign 
        = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
            >> 0x1fU) == (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
                          >> 0x1fU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
           - vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_21)
            ? 0U : ((0x63U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                     ? 3U : ((0xe3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                              ? 6U : ((0x263U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                       ? 2U : ((0x2e3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                ? 5U
                                                : (
                                                   (0x363U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                    ? 1U
                                                    : 
                                                   ((0x3e3U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                    << 2U)))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out 
        = ((0x37U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst))
            ? 0xbU : ((IData)(vlSelf->__VdfgTmp_h0634cb66__0)
                       ? 0U : ((0x113U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                ? 5U : ((0x193U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                         ? 7U : ((0x213U 
                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                  ? 4U
                                                  : 
                                                 ((0x313U 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                   ? 3U
                                                   : 
                                                  ((0x393U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                    ? 2U
                                                    : 
                                                   ((0x93U 
                                                     == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                     ? 6U
                                                     : 
                                                    ((0x293U 
                                                      == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                      ? 8U
                                                      : 
                                                     ((0x8293U 
                                                       == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                       ? 9U
                                                       : 
                                                      ((0x33U 
                                                        == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                        ? 0U
                                                        : 
                                                       ((0x8033U 
                                                         == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                         ? 1U
                                                         : 
                                                        ((0xb3U 
                                                          == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                          ? 6U
                                                          : 
                                                         ((0x133U 
                                                           == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                           ? 5U
                                                           : 
                                                          ((0x1b3U 
                                                            == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                            ? 7U
                                                            : 
                                                           ((0x233U 
                                                             == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                             ? 4U
                                                             : 
                                                            ((0x2b3U 
                                                              == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                              ? 8U
                                                              : 
                                                             ((0x82b3U 
                                                               == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                               ? 9U
                                                               : 
                                                              ((0x333U 
                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                                ? 3U
                                                                : 
                                                               ((0x3b3U 
                                                                 == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                                 ? 2U
                                                                 : 
                                                                ((IData)(vlSelf->__VdfgTmp_h8aef8152__0)
                                                                  ? 0xfU
                                                                  : 0xaU)))))))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0 
        = ((0x37U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
           | ((~ (IData)(vlSelf->__VdfgTmp_h0634cb66__0)) 
              & ((0x113U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x193U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x213U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       & ((0x313U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                          | ((0x393U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                             & ((0x93U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                & ((0x293U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                   & ((0x8293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                      | ((0x33U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                         & ((0x8033U 
                                             == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                            | ((0xb3U 
                                                != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                               & ((0x133U 
                                                   == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                  | ((0x1b3U 
                                                      == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                     | ((0x233U 
                                                         != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                        & ((0x2b3U 
                                                            != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                           & ((0x82b3U 
                                                               == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                              | ((0x333U 
                                                                  == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                 | ((0x3b3U 
                                                                     != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                    & (IData)(vlSelf->__VdfgTmp_h8aef8152__0)))))))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5 
        = ((0x100fU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22) 
            | ((0x57U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
               | ((0x2057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                  | ((0x1057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                     | ((0x3057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                        | ((0xd7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                           | ((0x2d7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                              | ((0x4d7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                 | ((0x457U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                    | ((0x557U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                       | ((0x267U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                          | ((0x2e7U 
                                              == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                             | ((0x1e7U 
                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                | ((0xe7U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                   | ((0x167U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                      | (0x367U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))))))))))))))))) 
           | (0x3e7U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16 
        = ((0x33U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
           | ((0x8033U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
              | ((0xb3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                 | ((0x133U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                    | ((0x1b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                       | ((0x233U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                          | ((0x2b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             | ((0x82b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                | (IData)(vlSelf->__VdfgTmp_hfd3c45f9__0)))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__bypass__io_idata_load 
        = ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift)
            : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type))
                ? (0xffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift)
                : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type))
                    ? (((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift 
                                          >> 7U)))) 
                        << 8U) | (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift))
                    : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type))
                        ? (((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift 
                                              >> 0xfU)))) 
                            << 0x10U) | (0xffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift))
                        : vlSelf->Tile__DOT___dcache_io_cpu_resp_bits_data))));
    vlSelf->io_nasti_w_bits_last = ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_1) 
                                    & (IData)(vlSelf->Tile__DOT__dcache__DOT__write_count));
    vlSelf->Tile__DOT__icache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
    vlSelf->Tile__DOT__dcache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
    vlSelf->Tile__DOT___dcache_io_nasti_ar_valid = 
        ((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
         & ((1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))
             ? ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__hit)) 
                & (~ (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty)))
             : ((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))
                 ? ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT___T_35)) 
                    & (~ (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty)))
                 : ((~ ((3U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                        | (4U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                    & (5U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))))));
    vlSelf->Tile__DOT___dcache_io_nasti_aw_valid = 
        ((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
         & ((1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))
             ? ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__hit)) 
                & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty))
             : ((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                & ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT___T_35)) 
                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT___T_1 
        = (((~ ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall))) 
            & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en)) 
           & (0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                              >> 7U))));
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)) 
                 & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
                    & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
                       | (1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15) 
           | ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13)))));
    if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt 
            = (1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
                     >> 0x1fU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu 
            = (1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
                     >> 0x1fU));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt 
            = (1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
                     >> 0x1fU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu 
            = (1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
                     >> 0x1fU));
    }
    vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0)
            ? 0U : ((0x23U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                     ? 3U : ((0xa3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                              ? 2U : (0x123U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)))));
    vlSelf->Tile__DOT___core_io_dcache_req_valid = 
        ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
         & (((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0)) 
             & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20)) 
            | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0)) 
               & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_17 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0 
        = ((0x13U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x113U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x193U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x213U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x313U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       | ((0x393U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                          | ((0x93U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             | ((0x293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                | ((0x8293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                   | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16))))))))));
    vlSelf->io_nasti_ar_bits_addr = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid)
                                      ? (0xfffffff0U 
                                         & vlSelf->Tile__DOT__dcache__DOT__addr_reg)
                                      : (0xfffffff0U 
                                         & vlSelf->Tile__DOT__icache__DOT__addr_reg));
    vlSelf->io_nasti_aw_valid = ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__arb__DOT__state)))) 
                                 & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_aw_valid));
    vlSelf->Tile__DOT__icache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en));
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___brCond_io_taken 
        = (((~ (IData)((0U != vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff))) 
            & (3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out))) 
           | (((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out)) 
               & (0U != vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff)) 
              | (((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out)) 
                  & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt)) 
                 | (((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt)) 
                     & (5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out))) 
                    | (((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out)) 
                        & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu)) 
                       | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu)) 
                          & (4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out))))))));
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en 
        = ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)) 
           & ((IData)(vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0) 
              & (IData)(vlSelf->Tile__DOT___core_io_dcache_req_valid)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
        = (((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0)) 
            & ((0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
               | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)) 
                  & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7) 
                     | ((0x23U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                        | ((0xa3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                           | ((0x123U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                              | ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_17) 
                                 | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15)) 
                                    & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14) 
                                       | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13)) 
                                          & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0))))))))))))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0
            : vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_pc);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0));
    vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)
            ? 3U : ((0x6fU == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst))
                     ? 4U : ((0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                              ? 1U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)
                                       ? 5U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7)
                                                ? 1U
                                                : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20)
                                                    ? 2U
                                                    : 
                                                   ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19)
                                                     ? 1U
                                                     : 
                                                    ((1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16) 
                                                         | ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75) 
                                                            | ((0x100fU 
                                                                == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                                                               | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3)) 
                                                                  | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14))))))
                                                      ? 0U
                                                      : 6U))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_5 
        = (0xfffU & ((2U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
                      ? ((0xfe0U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                    >> 0x14U)) | (0x1fU 
                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                     >> 7U)))
                      : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
                          ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                             >> 0x14U) : (0xffeU & 
                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                           >> 0x14U)))));
    vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0 
        = (1U & ((~ (IData)(vlSelf->io_nasti_aw_valid)) 
                 & (~ (IData)((0U != (IData)(vlSelf->Tile__DOT__arb__DOT__state))))));
    vlSelf->Tile__DOT__dcache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en));
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN 
        = ((0xccU & ((0xf0U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                               >> 4U)) | (0xcU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                  >> 0xcU)))) 
           | (0x33U & ((0xf0U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                 >> 8U)) | (0xfU & 
                                            (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                             >> 0x10U)))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_24 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75));
    vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel = 
        ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)
          ? 0U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9)
                   ? 1U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)
                            ? 0U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7)
                                     ? 2U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6)
                                              ? 0U : 
                                             ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5)
                                               ? 2U
                                               : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2)
                                                   ? 0U
                                                   : 
                                                  (3U 
                                                   & (- (IData)(
                                                                (0x10000073U 
                                                                 == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)))))))))));
    vlSelf->__VdfgTmp_hbd242f9a__0 = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0) 
                                      | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hdfe1f980__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0) 
           | ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75) 
              | ((0x100fU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                 | ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       | ((0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                          | ((0x2f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                             | ((0x373U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                | ((0x3f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                   | ((0x73U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                                      | ((0x100073U 
                                          == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                                         | ((0x10000073U 
                                             == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                                            | (0x10200073U 
                                               == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)))))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_7 
        = ((5U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
            ? ((0x1000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                           >> 0x13U)) | ((0x800U & 
                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                           << 4U)) 
                                         | ((0x7e0U 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                >> 0x14U)) 
                                            | (0x1eU 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                  >> 7U)))))
            : ((0x1000U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_5) 
                           << 1U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_5)));
    vlSelf->io_nasti_ar_valid = (((IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid) 
                                  | (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid)) 
                                 & (IData)(vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0));
    vlSelf->Tile__DOT___arb_io_dcache_ar_ready = ((IData)(vlSelf->io_nasti_ar_ready) 
                                                  & (IData)(vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 
        = (0x55555U & ((0x60000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                    << 0xdU)) | ((0x18000U 
                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                     << 9U)) 
                                                 | ((0x6000U 
                                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                        << 5U)) 
                                                    | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN) 
                                                        << 5U) 
                                                       | ((0x18U 
                                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                              >> 0xfU)) 
                                                          | ((6U 
                                                              & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                 >> 0x13U)) 
                                                             | (1U 
                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                   >> 0x17U)))))))));
    vlSelf->Tile__DOT__core__DOT___ctrl_io_inst_kill 
        = ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9) 
              | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)) 
                 & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7) 
                    | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6)) 
                       & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5) 
                          | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2)) 
                             & (0x10000073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out 
        = ((IData)(vlSelf->__VdfgTmp_hbd242f9a__0) ? 0xfU
            : ((0x433U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                ? 0U : ((0x4b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                         ? 1U : ((0x533U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                  ? 2U : ((0x5b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                           ? 3U : (
                                                   (0x633U 
                                                    == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                    ? 4U
                                                    : 
                                                   ((0x6b3U 
                                                     == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                     ? 5U
                                                     : 
                                                    ((0x733U 
                                                      == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)
                                                      ? 6U
                                                      : 
                                                     (7U 
                                                      | ((0x7b3U 
                                                          != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                         << 3U))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h5a0d4b50__0 
        = ((IData)(vlSelf->__VdfgTmp_hbd242f9a__0) 
           | ((0x433U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
              & ((0x4b3U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                 & ((0x533U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                    & ((0x5b3U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                       & ((0x633U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                          & ((0x6b3U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             & ((0x733U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                & (0x7b3U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)))))))));
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hdfe1f980__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___immGen_io_out 
        = ((6U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
            ? (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                        >> 0xfU)) : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
                                      ? (((- (IData)(
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                      >> 0x1fU))) 
                                          << 0x14U) 
                                         | ((0xff000U 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
                                            | ((0x800U 
                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                   >> 9U)) 
                                               | (0x7feU 
                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                                     >> 0x14U)))))
                                      : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
                                          ? (0xfffff000U 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)
                                          : (((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_7) 
                                                             >> 0xcU)))) 
                                              << 0xdU) 
                                             | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_7)))));
    vlSelf->Tile__DOT___arb_io_icache_ar_ready = ((~ (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid)) 
                                                  & (IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin 
        = (((0x37U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
            | ((~ (IData)(vlSelf->__VdfgTmp_h0634cb66__0)) 
               & ((0x113U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                  & ((0x193U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                     & ((0x213U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                        & ((0x313U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                           & ((0x393U != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                              & ((0x93U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                 & ((0x293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                    | ((0x8293U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                       | ((0x33U != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                          & ((0x8033U 
                                              != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                             & ((0xb3U 
                                                 != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                & ((0x133U 
                                                    != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                   & ((0x1b3U 
                                                       != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                      & ((0x233U 
                                                          != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                         & ((0x2b3U 
                                                             == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                            | ((~ (IData)(vlSelf->__VdfgTmp_hfd3c45f9__0)) 
                                                               | (0x82b3U 
                                                                  == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)))))))))))))))))))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
            : ((vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                << 0x1fU) | ((0x40000000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                             << 0x1dU)) 
                             | ((0x20000000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                << 0x1bU)) 
                                | ((0x10000000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                   << 0x19U)) 
                                   | ((0x8000000U & 
                                       (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        << 0x17U)) 
                                      | ((0x7800000U 
                                          & ((0xff800000U 
                                              & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 
                                                 << 8U)) 
                                             | (0x2800000U 
                                                & ((0x6000000U 
                                                    & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                       << 0x13U)) 
                                                   | (0x1800000U 
                                                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                         << 0xfU)))))) 
                                         | ((0x7f8000U 
                                             & ((0xffff8000U 
                                                 & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 
                                                    << 8U)) 
                                                | (0x2a8000U 
                                                   & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN) 
                                                      << 0xfU)))) 
                                            | ((0x4000U 
                                                & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN) 
                                                   << 0xdU)) 
                                               | ((0x2000U 
                                                   & ((0xffffe000U 
                                                       & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 
                                                          << 8U)) 
                                                      | (0x7ffe000U 
                                                         & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                            >> 5U)))) 
                                                  | ((0x1000U 
                                                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                         >> 7U)) 
                                                     | ((0x800U 
                                                         & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                            >> 9U)) 
                                                        | (((0x700U 
                                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 
                                                                << 8U)) 
                                                            | (0x280U 
                                                               & ((0x600U 
                                                                   & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                      >> 0xdU)) 
                                                                  | (0x180U 
                                                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                        >> 0x11U))))) 
                                                           | ((0x40U 
                                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                  >> 0x13U)) 
                                                              | ((0x20U 
                                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                     >> 0x15U)) 
                                                                 | ((0x10U 
                                                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                        >> 0x17U)) 
                                                                    | ((8U 
                                                                        & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                           >> 0x19U)) 
                                                                       | ((4U 
                                                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                              >> 0x1bU)) 
                                                                          | ((2U 
                                                                              & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                                >> 0x1dU)) 
                                                                             | (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                                                >> 0x1fU))))))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_4 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel)) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___brCond_io_taken));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0 
        = ((~ ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h5a0d4b50__0) 
               | ((7U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                  | ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                     | ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                        | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                           | (3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)))))))) 
           & (2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en 
        = (1U & (~ ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                     >> 0x19U) & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0)) 
                                  & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h990c992e__0)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0)
            ? 0U : ((0x57U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                     ? 1U : ((0x2057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                              ? 2U : ((0x1057U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                       ? 3U : ((0x3057U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                ? 4U
                                                : (
                                                   (0xd7U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                    ? 5U
                                                    : 
                                                   ((0x2d7U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                     ? 6U
                                                     : 
                                                    ((0x4d7U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                      ? 7U
                                                      : 
                                                     ((0x457U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                       ? 8U
                                                       : 
                                                      ((0x557U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                        ? 9U
                                                        : 
                                                       ((0x267U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                         ? 0xaU
                                                         : 
                                                        ((0x2e7U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                          ? 0xbU
                                                          : 
                                                         ((0x1e7U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                           ? 0xcU
                                                           : 
                                                          ((0xe7U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                            ? 0xdU
                                                            : 
                                                           ((0x167U 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                             ? 0xeU
                                                             : 
                                                            ((0x367U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1))
                                                              ? 0xfU
                                                              : 
                                                             ((0x3e7U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1)) 
                                                              << 4U)))))))))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
        = (((~ ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19))) 
            & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16) 
               | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18)) 
                  & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0))))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2
            : vlSelf->Tile__DOT__core__DOT__dpath__DOT___immGen_io_out);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel 
        = (((0x10U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
            | (0xfU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)))
            ? 4U : (((0xeU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                     | ((0xdU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                        | ((0xcU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                           | ((0xbU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                              | (0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))))))
                     ? 2U : (((9U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                              | ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                 | (7U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))))
                              ? 3U : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                      | ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                         | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                            | ((3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                               | ((2U 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                                  | (1U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT___GEN_0 
        = ((((QData)((IData)((- (IData)(((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0) 
                                         & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                            >> 0x1fU)))))) 
             << 0x20U) | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1))) 
           * (((QData)((IData)((- (IData)(((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0) 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                              >> 0x1fU)))))) 
               << 0x20U) | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
           + ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0)
               ? (- vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
               : vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2));
    if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type;
        vlSelf->__VdfgTmp_h92f16298__0 = (0x3fffffffU 
                                          & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
                                             >> 2U));
        vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
            = (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc);
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T 
            = vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type;
        vlSelf->__VdfgTmp_h92f16298__0 = (0x3fffffffU 
                                          & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
                                             >> 2U));
        vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
            = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)
                ? ((IData)(0x100U) + ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV) 
                                      << 6U)) : (((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)) 
                                                  & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9)) 
                                                     & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)) 
                                                        & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7)) 
                                                           & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6)) 
                                                              & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5)) 
                                                                 & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2)) 
                                                                    & (3U 
                                                                       == 
                                                                       (3U 
                                                                        & (- (IData)(
                                                                                (0x10000073U 
                                                                                == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst))))))))))))
                                                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc
                                                  : 
                                                 ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_4)
                                                   ? 
                                                  (0xfffffffeU 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum)
                                                   : 
                                                  ((2U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel))
                                                    ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc)
                                                    : 
                                                   ((IData)(4U) 
                                                    + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc))))));
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
        = (0x1ffffffffULL & VL_SHIFTRS_QQI(33,33,5, 
                                           (((QData)((IData)(
                                                             ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0) 
                                                              & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin 
                                                                 >> 0x1fU)))) 
                                             << 0x20U) 
                                            | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin))), 
                                           (0x1fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)));
    if ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0 
            = (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0 
            = (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0 
            = (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0 
            = (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op 
            = ((0x10U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                ? 2U : (0xfU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0 = 0U;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xdU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_hbe180063__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 9U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha0dc5a79__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 5U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h6d2b06e7__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 1U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    if ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type 
            = ((0xeU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                ? 5U : ((0xdU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                         ? 3U : ((0xcU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                  ? 1U : ((0xbU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                           ? 4U : (
                                                   (0xaU 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                                   << 1U)))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0 
            = (1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                     >> 0xcU));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0 = 0U;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h677344a9__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 0xdU))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7416c8a4__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
              >> 8U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h6743b8e4__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 9U))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7407838c__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
              >> 4U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h675c1f6c__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 5U))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h747068eb__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h66af7931__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 1U))));
    if ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
        vlSelf->__VdfgTmp_h6297395b__0 = ((9U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                           ? 4U : (
                                                   (8U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                                    ? 2U
                                                    : 
                                                   (7U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x18U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x1cU);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x18U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x1cU);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
    } else {
        vlSelf->__VdfgTmp_h6297395b__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0 = 0U;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xfU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xdU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xbU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 9U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 7U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 5U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 3U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 1U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x1fU));
    if ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0 
            = (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x1cU);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x18U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x1cU);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0 
            = (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0 
            = (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0 
            = (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op 
            = ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                ? 8U : ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                         ? 6U : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                  ? 3U : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                           ? 2U : (
                                                   (2U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                                    ? 1U
                                                    : 
                                                   ((1U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                                     ? 0U
                                                     : 0xaU))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0 
            = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x1fU);
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op = 0xaU;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0 = 0U;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN 
        = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
            >> 7U) & ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                         >> 0x1fU)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x1fU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
    vlSelf->Tile__DOT___core_io_dcache_req_bits_mask 
        = (0xfU & ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T))
                    ? ((IData)(1U) << (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum))
                    : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T))
                        ? ((IData)(3U) << (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum))
                        : (- (IData)((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T)))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_1 
        = ((0xccU & ((0xf0U & ((IData)((vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                        >> 8U)) << 4U)) 
                     | (0xcU & ((IData)((vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                         >> 0xeU)) 
                                << 2U)))) | (0x33U 
                                             & ((0xf0U 
                                                 & ((IData)(
                                                            (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                             >> 0xcU)) 
                                                    << 4U)) 
                                                | (0xfU 
                                                   & (IData)(
                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                              >> 0x10U))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 4U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                  >> 0x1cU) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 8U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                  >> 0x18U) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 4U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x1cU) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 8U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x18U) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 8U) | ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x18U) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN) 
            << 4U) | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                       ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                          >> 0x1cU) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN) 
            << 8U) | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                       ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                          >> 0x18U) : 0U));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0 
        = (0x7fffU & (- (IData)(((vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                  >> 7U) & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0)))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6 
        = (0x3ffU & (((0x3f0U & ((- (IData)(((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                >> 0x1bU)))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                     * ((0x3f0U & ((- (IData)(((3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                                  >> 0x1bU)))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4 
        = (0x3ffU & (((0x3f0U & ((- (IData)(((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                >> 0x13U)))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0)) 
                     * ((0x3f0U & ((- (IData)(((3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                                  >> 0x13U)))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2 
        = (0x3ffU & (((0x3f0U & ((- (IData)(((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                >> 0xbU)))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0)) 
                     * ((0x3f0U & ((- (IData)(((3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                                  >> 0xbU)))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0 
        = (0x3ffU & (((0x3f0U & ((- (IData)(((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                >> 3U)))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0)) 
                     * ((0x3f0U & ((- (IData)(((3U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                               & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                                  >> 3U)))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0))));
    vlSelf->Tile__DOT__dcache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->__VdfgTmp_h92f16298__0 
                                                  >> 2U)));
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                        >> 4U) : (vlSelf->__VdfgTmp_h92f16298__0 
                                  >> 2U)));
    vlSelf->Tile__DOT__icache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                                  >> 4U)));
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                        >> 4U) : (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                  >> 4U)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 
        = (0x55555U & ((0x60000U & ((IData)((vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                             >> 4U)) 
                                    << 0x11U)) | ((0x18000U 
                                                   & ((IData)(
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                               >> 6U)) 
                                                      << 0xfU)) 
                                                  | ((0x6000U 
                                                      & ((IData)(
                                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                  >> 8U)) 
                                                         << 0xdU)) 
                                                     | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_1) 
                                                         << 5U) 
                                                        | ((0x18U 
                                                            & ((IData)(
                                                                       (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                        >> 0x12U)) 
                                                               << 3U)) 
                                                           | ((6U 
                                                               & ((IData)(
                                                                          (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                           >> 0x14U)) 
                                                                  << 1U)) 
                                                              | (1U 
                                                                 & (IData)(
                                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                            >> 0x17U))))))))));
    if ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6 
            = ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2 
            = ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4 
            = ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6 
            = ((0x1f0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                           << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                     >> 0x17U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4 
            = ((0x1fff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                             << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                       >> 0xfU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6 
            = ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2));
    if ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2 
            = ((0x1fffff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                               << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                         >> 7U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0 
            = ((0xfffffff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                          << 1U))) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6 
            = ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2));
    if ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0 
            = ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0));
    if ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1 
        = (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1 
        = (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0) 
            << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0) 
            << 0x10U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0));
    if ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1 
            = ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0 
            = ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1 
        = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                      + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4 
        = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                      - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                    + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
    if ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))));
    } else if ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4));
    } else if ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1));
    } else if ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                          - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                        - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4));
    } else if ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
            = (0xffffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
            = (0xffffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
            = (0xffffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
            = (0xffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
            = (0xffffU & 0U);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
            = (0xffU & 0U);
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_out 
        = (((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)) 
            | (1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum
            : (((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)) 
                | (7U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)))
                ? (1U & (((vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                           >> 0x1fU) == (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                         >> 0x1fU))
                          ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
                             >> 0x1fU) : (((0x37U == 
                                            (0x7fU 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
                                           | ((~ (IData)(vlSelf->__VdfgTmp_h0634cb66__0)) 
                                              & ((0x113U 
                                                  != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                 & ((0x193U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                    | ((0x213U 
                                                        != (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                       & ((0x313U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                          | ((0x393U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                             | ((0x93U 
                                                                 == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                | ((0x293U 
                                                                    != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                   & ((0x8293U 
                                                                       != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                      & ((0x33U 
                                                                          != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                         & ((0x8033U 
                                                                             != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                            & ((0xb3U 
                                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                               | ((0x133U 
                                                                                != vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                                & ((~ 
                                                                                ((0x233U 
                                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                                | ((0x2b3U 
                                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                                                                | (0x82b3U 
                                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0)))) 
                                                                                | (0x1b3U 
                                                                                == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0))))))))))))))))
                                           ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                              >> 0x1fU)
                                           : (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                              >> 0x1fU))))
                : (((9U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)) 
                    | (8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out)))
                    ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5)
                    : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out))
                        ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5) 
                            << 0x1fU) | ((0x40000000U 
                                          & ((IData)(
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                      >> 1U)) 
                                             << 0x1eU)) 
                                         | ((0x20000000U 
                                             & ((IData)(
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                         >> 2U)) 
                                                << 0x1dU)) 
                                            | ((0x10000000U 
                                                & ((IData)(
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                            >> 3U)) 
                                                   << 0x1cU)) 
                                               | ((0x8000000U 
                                                   & ((IData)(
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                               >> 4U)) 
                                                      << 0x1bU)) 
                                                  | ((0x7800000U 
                                                      & ((0xff800000U 
                                                          & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 
                                                             << 8U)) 
                                                         | (0x2800000U 
                                                            & ((0x6000000U 
                                                                & ((IData)(
                                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                            >> 6U)) 
                                                                   << 0x19U)) 
                                                               | (0x1800000U 
                                                                  & ((IData)(
                                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                              >> 8U)) 
                                                                     << 0x17U)))))) 
                                                     | ((0x7f8000U 
                                                         & ((0xffff8000U 
                                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 
                                                                << 8U)) 
                                                            | (0x2a8000U 
                                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_1) 
                                                                  << 0xfU)))) 
                                                        | ((0x4000U 
                                                            & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_1) 
                                                               << 0xdU)) 
                                                           | ((0x2000U 
                                                               & ((0xffffe000U 
                                                                   & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 
                                                                      << 8U)) 
                                                                  | ((IData)(
                                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                              >> 0x12U)) 
                                                                     << 0xdU))) 
                                                              | ((0x1000U 
                                                                  & ((IData)(
                                                                             (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                              >> 0x13U)) 
                                                                     << 0xcU)) 
                                                                 | ((0x800U 
                                                                     & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x14U)) 
                                                                        << 0xbU)) 
                                                                    | (((0x700U 
                                                                         & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 
                                                                            << 8U)) 
                                                                        | (0x280U 
                                                                           & ((0x600U 
                                                                               & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x16U)) 
                                                                                << 9U)) 
                                                                              | (0x180U 
                                                                                & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x18U)) 
                                                                                << 7U))))) 
                                                                       | ((0x40U 
                                                                           & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x19U)) 
                                                                              << 6U)) 
                                                                          | ((0x20U 
                                                                              & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1aU)) 
                                                                                << 5U)) 
                                                                             | ((0x10U 
                                                                                & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1bU)) 
                                                                                << 4U)) 
                                                                                | ((8U 
                                                                                & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1cU)) 
                                                                                << 3U)) 
                                                                                | ((4U 
                                                                                & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1dU)) 
                                                                                << 2U)) 
                                                                                | ((2U 
                                                                                & ((IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1eU)) 
                                                                                << 1U)) 
                                                                                | (1U 
                                                                                & (IData)(
                                                                                (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
                                                                                >> 0x1fU)))))))))))))))))))))
                        : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out))
                            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                               & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
                            : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out))
                                ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                   | vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
                                : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out))
                                    ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                       ^ vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
                                    : ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out))
                                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
                                        : vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2))))))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3 
        = (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3 
        = (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2));
    if ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
            = (0x3ffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
            = (0x3ffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN);
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
            = (0x3ffffffU & (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
            = (0x3ffffU & (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2));
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b 
        = (0x3ffffU & ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
                           + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
                              + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
                                       + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
                                          + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
                                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7)))))))
                        : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
                           + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
                              + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
                                       + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
                                          + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
                                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7)))))))));
    if ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 
            = (0x7fffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 
            = (0x7fffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 
            = (0x7fffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 
            = (0x7fffffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11);
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 
            = (0x7fffffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1);
    } else if ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 
            = (0x7fffffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12 
                              << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 
            = (0x7fffffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13 
                              << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 
            = (0x7fffffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16 
                              << (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10 
                            << (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 
            = (0x7fffffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11 
                              << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 
            = (0x7fffffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1 
                            << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 
            = (0x7fffffffU & ((0x7fff0000U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55) 
                                                             >> 0xfU)))) 
                                              << 0x10U)) 
                              | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 
            = (0x7fffffffU & ((0x7fff0000U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85) 
                                                             >> 0xfU)))) 
                                              << 0x10U)) 
                              | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 
            = (0x7fffffffU & ((0x7fff0000U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115) 
                                                             >> 0xfU)))) 
                                              << 0x10U)) 
                              | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 
            = (0x7fffffffU & ((0x7fff0000U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25) 
                                                             >> 0xfU)))) 
                                              << 0x10U)) 
                              | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25)));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 
            = (0x7fffffU & ((0x7fff00U & ((- (IData)(
                                                     (1U 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25) 
                                                         >> 7U)))) 
                                          << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25)));
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5));
    if ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___br8x8_io_qvm_out 
            = ((0xc000U & ((((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                >> 0xeU)) + (2U & (
                                                   ((5U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                     ? 
                                                    (~ 
                                                     ((2U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                         >> 0xfU)))
                                                     : 
                                                    ((4U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                      ? 
                                                     VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7))
                                                      : 
                                                     ((2U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                       ? 
                                                      VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7))
                                                       : 
                                                      ((3U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                        ? 
                                                       ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7) 
                                                        != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7))
                                                        : 
                                                       ((1U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                        & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7) 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7))))))) 
                                                   << 1U))) 
                           << 0xeU)) | ((0x3000U & 
                                         (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0) 
                                           + (((5U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h677344a9__0)
                                                : (
                                                   (4U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                    ? 
                                                   VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6))
                                                    : 
                                                   ((2U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                     ? 
                                                    VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6))
                                                     : 
                                                    ((3U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                      ? 
                                                     ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6) 
                                                      != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6))
                                                      : 
                                                     ((1U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6) 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6))))))) 
                                              << 1U)) 
                                          << 0xcU)) 
                                        | ((0xc00U 
                                            & ((((2U 
                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                 & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                    >> 0xaU)) 
                                                + (2U 
                                                   & (((5U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                        ? 
                                                       (~ 
                                                        ((2U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                         & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                            >> 0xbU)))
                                                        : 
                                                       ((4U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                         ? 
                                                        VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5))
                                                         : 
                                                        ((2U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                          ? 
                                                         VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5))
                                                          : 
                                                         ((3U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                           ? 
                                                          ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5) 
                                                           != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5))
                                                           : 
                                                          ((1U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5) 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5))))))) 
                                                      << 1U))) 
                                               << 0xaU)) 
                                           | ((0x300U 
                                               & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7416c8a4__0) 
                                                   + 
                                                   (((5U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                      ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h6743b8e4__0)
                                                      : 
                                                     ((4U 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                       ? 
                                                      VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4))
                                                       : 
                                                      ((2U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                        ? 
                                                       VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4))
                                                        : 
                                                       ((3U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                         ? 
                                                        ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4) 
                                                         != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4))
                                                         : 
                                                        ((1U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                         & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4) 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4))))))) 
                                                    << 1U)) 
                                                  << 8U)) 
                                              | ((0xc0U 
                                                  & ((((2U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                       & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                          >> 6U)) 
                                                      + 
                                                      (2U 
                                                       & (((5U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                            ? 
                                                           (~ 
                                                            ((2U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                >> 7U)))
                                                            : 
                                                           ((4U 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                             ? 
                                                            VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3))
                                                             : 
                                                            ((2U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                              ? 
                                                             VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3))
                                                              : 
                                                             ((3U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                               ? 
                                                              ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3) 
                                                               != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3))
                                                               : 
                                                              ((1U 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3) 
                                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3))))))) 
                                                          << 1U))) 
                                                     << 6U)) 
                                                 | ((0x30U 
                                                     & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7407838c__0) 
                                                         + 
                                                         (((5U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h675c1f6c__0)
                                                            : 
                                                           ((4U 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                             ? 
                                                            VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2))
                                                             : 
                                                            ((2U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                              ? 
                                                             VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2))
                                                              : 
                                                             ((3U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                               ? 
                                                              ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2) 
                                                               != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2))
                                                               : 
                                                              ((1U 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2) 
                                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2))))))) 
                                                          << 1U)) 
                                                        << 4U)) 
                                                    | ((0xcU 
                                                        & ((((2U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                >> 2U)) 
                                                            + 
                                                            (2U 
                                                             & (((5U 
                                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                  ? 
                                                                 (~ 
                                                                  ((2U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                   & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                      >> 3U)))
                                                                  : 
                                                                 ((4U 
                                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                   ? 
                                                                  VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1))
                                                                   : 
                                                                  ((2U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                    ? 
                                                                   VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1))
                                                                    : 
                                                                   ((3U 
                                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                     ? 
                                                                    ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1) 
                                                                     != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1))
                                                                     : 
                                                                    ((1U 
                                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1) 
                                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1))))))) 
                                                                << 1U))) 
                                                           << 2U)) 
                                                       | (3U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h747068eb__0) 
                                                             + 
                                                             (((5U 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h66af7931__0)
                                                                : 
                                                               ((4U 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                 ? 
                                                                VL_GTES_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0))
                                                                 : 
                                                                ((2U 
                                                                  == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                  ? 
                                                                 VL_LTS_III(8, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0))
                                                                  : 
                                                                 ((3U 
                                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                                   ? 
                                                                  ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0) 
                                                                   != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0))
                                                                   : 
                                                                  ((1U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                                   & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0) 
                                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0))))))) 
                                                              << 1U))))))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 
            = (0x3ffffffU & ((0x3fc0000U & ((- (IData)(
                                                       (1U 
                                                        & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b 
                                                           >> 0x11U)))) 
                                            << 0x12U)) 
                             | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___br8x8_io_qvm_out 
            = ((0x7000U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0) 
                            + (((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                 ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h677344a9__0)
                                 : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                     ? VL_GTES_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3))
                                     : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                         ? VL_LTS_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3))
                                         : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3) 
                                                != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3))
                                             : ((1U 
                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3) 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3))))))) 
                               << 1U)) << 0xcU)) | 
               ((0x700U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7416c8a4__0) 
                            + (((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                 ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h6743b8e4__0)
                                 : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                     ? VL_GTES_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2))
                                     : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                         ? VL_LTS_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2))
                                         : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2) 
                                                != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2))
                                             : ((1U 
                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2) 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2))))))) 
                               << 1U)) << 8U)) | ((0x70U 
                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7407838c__0) 
                                                       + 
                                                       (((5U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                          ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h675c1f6c__0)
                                                          : 
                                                         ((4U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                           ? 
                                                          VL_GTES_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1))
                                                           : 
                                                          ((2U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                            ? 
                                                           VL_LTS_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1))
                                                            : 
                                                           ((3U 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                             ? 
                                                            ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1) 
                                                             != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1))
                                                             : 
                                                            ((1U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1) 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1))))))) 
                                                        << 1U)) 
                                                      << 4U)) 
                                                  | (7U 
                                                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h747068eb__0) 
                                                        + 
                                                        (((5U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                           ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h66af7931__0)
                                                           : 
                                                          ((4U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                            ? 
                                                           VL_GTES_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0))
                                                            : 
                                                           ((2U 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                             ? 
                                                            VL_LTS_III(16, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0))
                                                             : 
                                                            ((3U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type))
                                                              ? 
                                                             ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0) 
                                                              != (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0))
                                                              : 
                                                             ((1U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type)) 
                                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0) 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0))))))) 
                                                         << 1U))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 
            = (0x3ffffffU & ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
                                       + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3)))
                              : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
                                       + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3)))));
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mul8x8_io_out 
        = ((4U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
            ? (((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 
                                  >> 0x19U)))) << 0x1aU) 
               | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3)
            : ((2U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
                ? ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                    ? ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36)
                          ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7
                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7)
                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                        << 0x1cU) | ((0xf000000U & 
                                      (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                                         ? ((0x80U 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                             ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6
                                             : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6)
                                         : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                                       << 0x18U)) | 
                                     ((0xf00000U & 
                                       (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34)
                                          ? ((0x80U 
                                              & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5
                                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5)
                                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                                        << 0x14U)) 
                                      | ((0xf0000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4
                                                   : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4)
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0)) 
                                             << 0x10U)) 
                                         | ((0xf000U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3
                                                   : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3)
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                                                << 0xcU)) 
                                            | ((0xf00U 
                                                & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2
                                                      : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2)
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0)) 
                                                   << 8U)) 
                                               | ((0xf0U 
                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30)
                                                        ? 
                                                       ((0x80U 
                                                         & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1
                                                         : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1)
                                                        : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                                                      << 4U)) 
                                                  | (0xfU 
                                                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                         ? 
                                                        ((0x80U 
                                                          & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0
                                                          : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0)
                                                         : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0))))))))))
                    : ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                          ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3
                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3)
                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                        << 0x18U) | ((0xff0000U & (
                                                   ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2
                                                      : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2)
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                                                   << 0x10U)) 
                                     | ((0xff00U & 
                                         (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                            ? ((0x80U 
                                                & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1
                                                : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1)
                                            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                                          << 8U)) | 
                                        (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                   ? 
                                                  ((0x80U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0
                                                    : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0)
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)))))))
                : ((1U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
                    ? ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                        ? ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36)
                              ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 
                                     >> 4U) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 
                                               >> 4U))
                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                            << 0x1cU) | ((0xf000000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
                                                   >> 4U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
                                                   >> 4U))
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                                             << 0x18U)) 
                                         | ((0xf00000U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
                                                   >> 4U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
                                                   >> 4U))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                                                << 0x14U)) 
                                            | ((0xf0000U 
                                                & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
                                                      >> 4U)
                                                      : 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
                                                      >> 4U))
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0)) 
                                                   << 0x10U)) 
                                               | ((0xf000U 
                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32)
                                                        ? 
                                                       ((0x80U 
                                                         & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                         ? 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
                                                         >> 4U)
                                                         : 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
                                                         >> 4U))
                                                        : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                                                      << 0xcU)) 
                                                  | ((0xf00U 
                                                      & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                           ? 
                                                          ((0x80U 
                                                            & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                            ? 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
                                                            >> 4U)
                                                            : 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
                                                            >> 4U))
                                                           : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0)) 
                                                         << 8U)) 
                                                     | ((0xf0U 
                                                         & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30)
                                                              ? 
                                                             ((0x80U 
                                                               & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                               ? 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
                                                               >> 4U)
                                                               : 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
                                                               >> 4U))
                                                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                                                            << 4U)) 
                                                        | (0xfU 
                                                           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                               ? 
                                                              ((0x80U 
                                                                & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                                ? 
                                                               (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
                                                                >> 4U)
                                                                : 
                                                               (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
                                                                >> 4U))
                                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0))))))))))
                        : ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                              ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 
                                     >> 8U) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 
                                               >> 8U))
                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                            << 0x18U) | ((0xff0000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
                                                   >> 8U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
                                                   >> 8U))
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                                             << 0x10U)) 
                                         | ((0xff00U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
                                                   >> 8U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
                                                   >> 8U))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                                                << 8U)) 
                                            | (0xffU 
                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                   ? 
                                                  ((0x80U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
                                                    >> 8U)
                                                    : 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
                                                    >> 8U))
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)))))))
                    : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
                        : 0U))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6));
    if ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8 
            = (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4 
            = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7));
    } else if ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8 
            = (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4 
            = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4 = 0U;
    }
    if ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___alu8x8_io_out 
            = ((((1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                            & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                               >> 0xfU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)))
                  ? ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7
                      : ((0x7fffffffU & VL_SHIFTRS_III(31,31,5, 
                                                       (0x7fffffffU 
                                                        & (((0x7f800000U 
                                                             & ((- (IData)(
                                                                           (1U 
                                                                            & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 
                                                                               >> 0x16U)))) 
                                                                << 0x17U)) 
                                                            | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7) 
                                                           * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                       (0x1fU 
                                                        & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                           >> 0x11U)))) 
                         + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                            >> 9U))) : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                << 0x1cU) | ((0xf000000U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0)
                                              ? ((0x100U 
                                                  & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6
                                                  : 
                                                 ((0x7fffffffU 
                                                   & VL_SHIFTRS_III(31,31,5, 
                                                                    (0x7fffffffU 
                                                                     & (((0x7f800000U 
                                                                          & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 
                                                                                >> 0x16U)))) 
                                                                             << 0x17U)) 
                                                                         | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6) 
                                                                        * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                    (0x1fU 
                                                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                        >> 0x11U)))) 
                                                  + 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                   >> 9U)))
                                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                                            << 0x18U)) 
                             | ((0xf00000U & (((1U 
                                                & ((~ 
                                                    ((1U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                        >> 0xbU))) 
                                                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)))
                                                ? (
                                                   (0x100U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5
                                                    : 
                                                   ((0x7fffffffU 
                                                     & VL_SHIFTRS_III(31,31,5, 
                                                                      (0x7fffffffU 
                                                                       & (((0x7f800000U 
                                                                            & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 
                                                                                >> 0x16U)))) 
                                                                               << 0x17U)) 
                                                                           | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5) 
                                                                          * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                      (0x1fU 
                                                                       & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                          >> 0x11U)))) 
                                                    + 
                                                    (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                     >> 9U)))
                                                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0)) 
                                              << 0x14U)) 
                                | ((0xf0000U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_hbe180063__0)
                                                  ? 
                                                 ((0x100U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4
                                                   : 
                                                  ((0x7fffffffU 
                                                    & VL_SHIFTRS_III(31,31,5, 
                                                                     (0x7fffffffU 
                                                                      & (((0x7f800000U 
                                                                           & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 
                                                                                >> 0x16U)))) 
                                                                              << 0x17U)) 
                                                                          | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4) 
                                                                         * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                     (0x1fU 
                                                                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                         >> 0x11U)))) 
                                                   + 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                    >> 9U)))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0)) 
                                                << 0x10U)) 
                                   | ((0xf000U & ((
                                                   (1U 
                                                    & ((~ 
                                                        ((1U 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                         & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                            >> 7U))) 
                                                       | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)))
                                                    ? 
                                                   ((0x100U 
                                                     & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                     ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3
                                                     : 
                                                    ((0x7fffffffU 
                                                      & VL_SHIFTRS_III(31,31,5, 
                                                                       (0x7fffffffU 
                                                                        & (((0x7f800000U 
                                                                             & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 
                                                                                >> 0x16U)))) 
                                                                                << 0x17U)) 
                                                                            | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3) 
                                                                           * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                       (0x1fU 
                                                                        & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                           >> 0x11U)))) 
                                                     + 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                      >> 9U)))
                                                    : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0)) 
                                                  << 0xcU)) 
                                      | ((0xf00U & 
                                          (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha0dc5a79__0)
                                             ? ((0x100U 
                                                 & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                 ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2
                                                 : 
                                                ((0x7fffffffU 
                                                  & VL_SHIFTRS_III(31,31,5, 
                                                                   (0x7fffffffU 
                                                                    & (((0x7f800000U 
                                                                         & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 
                                                                                >> 0x16U)))) 
                                                                            << 0x17U)) 
                                                                        | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2) 
                                                                       * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                   (0x1fU 
                                                                    & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                       >> 0x11U)))) 
                                                 + 
                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                  >> 9U)))
                                             : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0)) 
                                           << 8U)) 
                                         | ((0xf0U 
                                             & (((1U 
                                                  & ((~ 
                                                      ((1U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                       & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                          >> 3U))) 
                                                     | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)))
                                                  ? 
                                                 ((0x100U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1
                                                   : 
                                                  ((0x7fffffffU 
                                                    & VL_SHIFTRS_III(31,31,5, 
                                                                     (0x7fffffffU 
                                                                      & (((0x7f800000U 
                                                                           & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 
                                                                                >> 0x16U)))) 
                                                                              << 0x17U)) 
                                                                          | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1) 
                                                                         * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                     (0x1fU 
                                                                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                         >> 0x11U)))) 
                                                   + 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                    >> 9U)))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0)) 
                                                << 4U)) 
                                            | (0xfU 
                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h6d2b06e7__0)
                                                   ? 
                                                  ((0x100U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0
                                                    : 
                                                   ((0x7fffffffU 
                                                     & VL_SHIFTRS_III(31,31,5, 
                                                                      (0x7fffffffU 
                                                                       & (((0x7f800000U 
                                                                            & ((- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 
                                                                                >> 0x16U)))) 
                                                                               << 0x17U)) 
                                                                           | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0) 
                                                                          * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17)), 
                                                                      (0x1fU 
                                                                       & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                          >> 0x11U)))) 
                                                    + 
                                                    (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                     >> 9U)))
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0))))))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3 
            = ((0x1e0U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                            >> 4U)))) 
                          << 5U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4));
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___alu8x8_io_out 
            = ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0)
                  ? ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3
                      : ((IData)((0x7fffffffffULL & 
                                  VL_SHIFTRS_QQI(39,39,5, 
                                                 (0x7fffffffffULL 
                                                  & ((((QData)((IData)(
                                                                       (0xffU 
                                                                        & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 
                                                                                >> 0x1eU))))))) 
                                                       << 0x1fU) 
                                                      | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3))) 
                                                     * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19)), 
                                                 (0x1fU 
                                                  & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                     >> 0x11U))))) 
                         + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                            >> 9U))) : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0)) 
                << 0x18U) | ((0xff0000U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_hbe180063__0)
                                             ? ((0x100U 
                                                 & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                 ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2
                                                 : 
                                                ((IData)(
                                                         (0x7fffffffffULL 
                                                          & VL_SHIFTRS_QQI(39,39,5, 
                                                                           (0x7fffffffffULL 
                                                                            & ((((QData)((IData)(
                                                                                (0xffU 
                                                                                & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 
                                                                                >> 0x1eU))))))) 
                                                                                << 0x1fU) 
                                                                                | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2))) 
                                                                               * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19)), 
                                                                           (0x1fU 
                                                                            & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                               >> 0x11U))))) 
                                                 + 
                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                  >> 9U)))
                                             : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0)) 
                                           << 0x10U)) 
                             | ((0xff00U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha0dc5a79__0)
                                              ? ((0x100U 
                                                  & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1
                                                  : 
                                                 ((IData)(
                                                          (0x7fffffffffULL 
                                                           & VL_SHIFTRS_QQI(39,39,5, 
                                                                            (0x7fffffffffULL 
                                                                             & ((((QData)((IData)(
                                                                                (0xffU 
                                                                                & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 
                                                                                >> 0x1eU))))))) 
                                                                                << 0x1fU) 
                                                                                | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1))) 
                                                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19)), 
                                                                            (0x1fU 
                                                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                                >> 0x11U))))) 
                                                  + 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                   >> 9U)))
                                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0)) 
                                            << 8U)) 
                                | (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h6d2b06e7__0)
                                             ? ((0x100U 
                                                 & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                 ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0
                                                 : 
                                                ((IData)(
                                                         (0x7fffffffffULL 
                                                          & VL_SHIFTRS_QQI(39,39,5, 
                                                                           (0x7fffffffffULL 
                                                                            & ((((QData)((IData)(
                                                                                (0xffU 
                                                                                & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 
                                                                                >> 0x1eU))))))) 
                                                                                << 0x1fU) 
                                                                                | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0))) 
                                                                               * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19)), 
                                                                           (0x1fU 
                                                                            & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                                               >> 0x11U))))) 
                                                 + 
                                                 (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                  >> 9U)))
                                             : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8;
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mm8x8_io_qvm_out 
        = ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
            ? 0U : ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                     ? ((0xc000U & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                                      ? (2U | ((4U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                   >> 0xeU)
                                                : 0U))
                                      : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                         & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                            >> 0xeU))) 
                                    << 0xeU)) | ((0x3000U 
                                                  & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
                                                       ? 
                                                      (2U 
                                                       | ((4U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                           ? 
                                                          ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                           >> 0xcU)
                                                           : 0U))
                                                       : 
                                                      ((4U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                       & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                          >> 0xcU))) 
                                                     << 0xcU)) 
                                                 | ((0xc00U 
                                                     & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
                                                          ? 
                                                         (2U 
                                                          | ((4U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                              ? 
                                                             ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                              >> 0xaU)
                                                              : 0U))
                                                          : 
                                                         ((4U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                             >> 0xaU))) 
                                                        << 0xaU)) 
                                                    | ((0x300U 
                                                        & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
                                                             ? 
                                                            (2U 
                                                             | ((4U 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                 ? 
                                                                ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                 >> 8U)
                                                                 : 0U))
                                                             : 
                                                            ((4U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                >> 8U))) 
                                                           << 8U)) 
                                                       | ((0xc0U 
                                                           & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
                                                                ? 
                                                               (2U 
                                                                | ((4U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                    ? 
                                                                   ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                    >> 6U)
                                                                    : 0U))
                                                                : 
                                                               ((4U 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                   >> 6U))) 
                                                              << 6U)) 
                                                          | ((0x30U 
                                                              & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
                                                                   ? 
                                                                  (2U 
                                                                   | ((4U 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                       ? 
                                                                      ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                       >> 4U)
                                                                       : 0U))
                                                                   : 
                                                                  ((4U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                   & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                      >> 4U))) 
                                                                 << 4U)) 
                                                             | ((0xcU 
                                                                 & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
                                                                      ? 
                                                                     (2U 
                                                                      | ((4U 
                                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                          ? 
                                                                         ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                          >> 2U)
                                                                          : 0U))
                                                                      : 
                                                                     ((4U 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                         >> 2U))) 
                                                                    << 2U)) 
                                                                | (3U 
                                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0))
                                                                       ? 
                                                                      (2U 
                                                                       | ((4U 
                                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                           ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in)
                                                                           : 0U))
                                                                       : 
                                                                      ((4U 
                                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                       & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in)))))))))))
                     : (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                           ? (2U | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0))
                           : (0xdU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0))) 
                         << 0xcU) | (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
                                        ? (2U | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0))
                                        : (0xdU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0))) 
                                      << 8U) | (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
                                                   ? 
                                                  (2U 
                                                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0))
                                                   : 
                                                  (0xdU 
                                                   & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0))) 
                                                 << 4U) 
                                                | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0))
                                                    ? 
                                                   (2U 
                                                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0))
                                                    : 
                                                   (0xdU 
                                                    & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0))))))));
}

VL_ATTR_COLD void VTile___024root___eval_stl(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_stl\n"); );
    // Body
    if (vlSelf->__VstlTriggered.at(0U)) {
        VTile___024root___stl_sequent__TOP__0(vlSelf);
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__ico(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if (vlSelf->__VicoTriggered.at(0U)) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__act(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if (vlSelf->__VactTriggered.at(0U)) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clock)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__nba(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if (vlSelf->__VnbaTriggered.at(0U)) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clock)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void VTile___024root___ctor_var_reset(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clock = VL_RAND_RESET_I(1);
    vlSelf->reset = VL_RAND_RESET_I(1);
    vlSelf->io_host_fromhost_valid = VL_RAND_RESET_I(1);
    vlSelf->io_host_fromhost_bits = VL_RAND_RESET_I(32);
    vlSelf->io_nasti_aw_ready = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_w_ready = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_b_valid = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_b_bits_id = VL_RAND_RESET_I(5);
    vlSelf->io_nasti_b_bits_resp = VL_RAND_RESET_I(2);
    vlSelf->io_nasti_ar_ready = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_r_valid = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_r_bits_id = VL_RAND_RESET_I(5);
    vlSelf->io_nasti_r_bits_data = VL_RAND_RESET_Q(64);
    vlSelf->io_nasti_r_bits_resp = VL_RAND_RESET_I(2);
    vlSelf->io_nasti_r_bits_last = VL_RAND_RESET_I(1);
    vlSelf->io_host_tohost = VL_RAND_RESET_I(32);
    vlSelf->io_nasti_aw_valid = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_aw_bits_id = VL_RAND_RESET_I(5);
    vlSelf->io_nasti_aw_bits_addr = VL_RAND_RESET_I(32);
    vlSelf->io_nasti_aw_bits_len = VL_RAND_RESET_I(8);
    vlSelf->io_nasti_aw_bits_size = VL_RAND_RESET_I(3);
    vlSelf->io_nasti_aw_bits_burst = VL_RAND_RESET_I(2);
    vlSelf->io_nasti_aw_bits_lock = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_aw_bits_cache = VL_RAND_RESET_I(4);
    vlSelf->io_nasti_aw_bits_prot = VL_RAND_RESET_I(3);
    vlSelf->io_nasti_aw_bits_qos = VL_RAND_RESET_I(4);
    vlSelf->io_nasti_w_valid = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_w_bits_data = VL_RAND_RESET_Q(64);
    vlSelf->io_nasti_w_bits_strb = VL_RAND_RESET_I(8);
    vlSelf->io_nasti_w_bits_last = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_b_ready = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_ar_valid = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_ar_bits_id = VL_RAND_RESET_I(5);
    vlSelf->io_nasti_ar_bits_addr = VL_RAND_RESET_I(32);
    vlSelf->io_nasti_ar_bits_len = VL_RAND_RESET_I(8);
    vlSelf->io_nasti_ar_bits_size = VL_RAND_RESET_I(3);
    vlSelf->io_nasti_ar_bits_burst = VL_RAND_RESET_I(2);
    vlSelf->io_nasti_ar_bits_lock = VL_RAND_RESET_I(1);
    vlSelf->io_nasti_ar_bits_cache = VL_RAND_RESET_I(4);
    vlSelf->io_nasti_ar_bits_prot = VL_RAND_RESET_I(3);
    vlSelf->io_nasti_ar_bits_qos = VL_RAND_RESET_I(4);
    vlSelf->io_nasti_r_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___arb_io_icache_ar_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___arb_io_dcache_aw_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___arb_io_dcache_w_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___arb_io_dcache_ar_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___dcache_io_cpu_resp_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___dcache_io_cpu_resp_bits_data = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT___dcache_io_nasti_aw_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___dcache_io_nasti_w_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___dcache_io_nasti_b_ready = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___dcache_io_nasti_ar_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___icache_io_cpu_resp_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___icache_io_nasti_ar_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___core_io_icache_req_bits_addr = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT___core_io_dcache_req_valid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT___core_io_dcache_req_bits_mask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT___ctrl_io_inst_kill = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_alu_op_out = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_br_op_out = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___brCond_io_taken = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___immGen_io_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_pc = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__illegal = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc_check = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__started = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc = VL_RAND_RESET_Q(33);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_1 = VL_RAND_RESET_Q(33);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_4 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__bypass__io_idata_load = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 = VL_RAND_RESET_I(2);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIP = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIE = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIP = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIE = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtimecmp = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mscratch = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mcause = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mbadaddr = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mfromhost = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qstate = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT___io_expt_T_12 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__isInstRet = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_24 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_25 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__unnamedblk3__DOT___GEN_0 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT___T_1 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_0 = VL_RAND_RESET_I(19);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 = VL_RAND_RESET_Q(33);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN_2 = VL_RAND_RESET_I(19);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_5 = VL_RAND_RESET_I(12);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__immGen__DOT___io_out_T_7 = VL_RAND_RESET_I(13);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT___GEN_0 = VL_RAND_RESET_Q(64);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h5a0d4b50__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__bypass__DOT____VdfgTmp_h8e44abc1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mm8x8_io_qvm_out = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___br8x8_io_qvm_out = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___alu8x8_io_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mul8x8_io_out = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7 = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3 = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b = VL_RAND_RESET_I(18);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 = VL_RAND_RESET_I(26);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 = VL_RAND_RESET_I(23);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17 = VL_RAND_RESET_I(31);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19 = VL_RAND_RESET_Q(39);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_hbe180063__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha0dc5a79__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h6d2b06e7__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3 = VL_RAND_RESET_I(16);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h677344a9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7416c8a4__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h6743b8e4__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7407838c__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h675c1f6c__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h747068eb__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h66af7931__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4 = VL_RAND_RESET_I(5);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3 = VL_RAND_RESET_I(9);
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0 = 0;
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN = VL_RAND_RESET_I(10);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0 = VL_RAND_RESET_I(17);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1 = VL_RAND_RESET_I(15);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_17 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_21 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_24 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h990c992e__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hdfe1f980__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0 = 0;
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0 = 0;
    vlSelf->Tile__DOT__icache__DOT__writeEnable = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__icache__DOT__wdata);
    vlSelf->Tile__DOT__icache__DOT__hit = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__state = VL_RAND_RESET_I(3);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__icache__DOT__v);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__icache__DOT__d);
    vlSelf->Tile__DOT__icache__DOT___GEN_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__icache__DOT___GEN_1 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__addr_reg = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__icache__DOT__cpu_data = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__icache__DOT__cpu_mask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__icache__DOT___T_22 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__read_count = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__read_wrap_out = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__is_alloc = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__is_alloc_reg = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__ren_reg = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__icache__DOT__rdata);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__icache__DOT__rdata_buf);
    vlSelf->Tile__DOT__icache__DOT__refill_buf_0 = VL_RAND_RESET_Q(64);
    vlSelf->Tile__DOT__icache__DOT__refill_buf_1 = VL_RAND_RESET_Q(64);
    vlSelf->Tile__DOT__icache__DOT___wmask_T_2 = VL_RAND_RESET_I(19);
    vlSelf->Tile__DOT__icache__DOT__is_dirty = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT___T_35 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask = VL_RAND_RESET_I(4);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1);
    vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5 = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 = VL_RAND_RESET_I(24);
    vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(20);
    }
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_0 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    vlSelf->Tile__DOT__dcache__DOT__writeEnable = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__dcache__DOT__wdata);
    vlSelf->Tile__DOT__dcache__DOT__hit = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata = VL_RAND_RESET_I(20);
    vlSelf->Tile__DOT__dcache__DOT__state = VL_RAND_RESET_I(3);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__dcache__DOT__v);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__dcache__DOT__d);
    vlSelf->Tile__DOT__dcache__DOT___GEN_0 = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__dcache__DOT___GEN_1 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__addr_reg = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__dcache__DOT__cpu_data = VL_RAND_RESET_I(32);
    vlSelf->Tile__DOT__dcache__DOT__cpu_mask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__dcache__DOT___T_22 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__read_count = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__read_wrap_out = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT___T_1 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__write_count = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__is_alloc = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT___T_34 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__ren_reg = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__dcache__DOT__rdata);
    VL_RAND_RESET_W(128, vlSelf->Tile__DOT__dcache__DOT__rdata_buf);
    vlSelf->Tile__DOT__dcache__DOT__refill_buf_0 = VL_RAND_RESET_Q(64);
    vlSelf->Tile__DOT__dcache__DOT__refill_buf_1 = VL_RAND_RESET_Q(64);
    vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 = VL_RAND_RESET_I(19);
    vlSelf->Tile__DOT__dcache__DOT__is_dirty = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT___T_35 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT___GEN_4 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask = VL_RAND_RESET_I(4);
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask = VL_RAND_RESET_I(4);
    VL_RAND_RESET_W(256, vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1);
    vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5 = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 = VL_RAND_RESET_I(24);
    vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(20);
    }
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_0 = VL_RAND_RESET_I(1);
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN = VL_RAND_RESET_I(8);
    vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory[__Vi0] = VL_RAND_RESET_I(32);
    }
    vlSelf->Tile__DOT__arb__DOT__state = VL_RAND_RESET_I(3);
    vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0 = 0;
    vlSelf->__VdfgTmp_h92f16298__0 = 0;
    vlSelf->__VdfgTmp_h0634cb66__0 = 0;
    vlSelf->__VdfgTmp_h8aef8152__0 = 0;
    vlSelf->__VdfgTmp_hfd3c45f9__0 = 0;
    vlSelf->__VdfgTmp_hbd242f9a__0 = 0;
    vlSelf->__VdfgTmp_h6297395b__0 = 0;
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc = VL_RAND_RESET_Q(33);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth = VL_RAND_RESET_I(32);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV = VL_RAND_RESET_I(2);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE = VL_RAND_RESET_I(1);
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 = VL_RAND_RESET_I(1);
    vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 = 0;
    vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 = VL_RAND_RESET_I(32);
    vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 = 0;
    vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 = 0;
    vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 = VL_RAND_RESET_I(16);
    vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 = 0;
    vlSelf->__Vdly__Tile__DOT__icache__DOT__read_count = VL_RAND_RESET_I(1);
    vlSelf->__Vdly__Tile__DOT__icache__DOT__state = VL_RAND_RESET_I(3);
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__read_count = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(256, vlSelf->__Vdly__Tile__DOT__dcache__DOT__v);
    VL_RAND_RESET_W(256, vlSelf->__Vdly__Tile__DOT__dcache__DOT__d);
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__state = VL_RAND_RESET_I(3);
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    vlSelf->__Vdly__Tile__DOT__arb__DOT__state = VL_RAND_RESET_I(3);
    vlSelf->__Vtrigrprev__TOP__clock = VL_RAND_RESET_I(1);
}
