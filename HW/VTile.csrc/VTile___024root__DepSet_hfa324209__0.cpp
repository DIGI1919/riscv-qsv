// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VTile.h for the primary calling header

#include "verilated.h"

#include "VTile___024root.h"

VL_INLINE_OPT void VTile___024root___ico_sequent__TOP__0(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->Tile__DOT___arb_io_dcache_aw_ready = ((~ (IData)(
                                                             (0U 
                                                              != (IData)(vlSelf->Tile__DOT__arb__DOT__state)))) 
                                                  & (IData)(vlSelf->io_nasti_aw_ready));
    vlSelf->Tile__DOT___arb_io_dcache_ar_ready = ((IData)(vlSelf->io_nasti_ar_ready) 
                                                  & (IData)(vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0));
    vlSelf->Tile__DOT___arb_io_dcache_w_ready = ((IData)(vlSelf->io_nasti_w_ready) 
                                                 & (3U 
                                                    == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
    vlSelf->Tile__DOT__icache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (1U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT__dcache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (2U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT___arb_io_icache_ar_ready = ((~ (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid)) 
                                                  & (IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready));
    vlSelf->Tile__DOT__dcache__DOT___T_1 = ((IData)(vlSelf->Tile__DOT___arb_io_dcache_w_ready) 
                                            & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid));
    vlSelf->Tile__DOT__icache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__icache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__read_count));
    vlSelf->Tile__DOT__dcache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_count));
    vlSelf->io_nasti_w_bits_last = ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_1) 
                                    & (IData)(vlSelf->Tile__DOT__dcache__DOT__write_count));
    vlSelf->Tile__DOT__icache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__icache__DOT__read_wrap_out));
    vlSelf->Tile__DOT__dcache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_wrap_out));
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
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
            & (IData)(vlSelf->Tile__DOT__icache__DOT___T_35)) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
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
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
            & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)) 
               & (IData)(vlSelf->Tile__DOT__dcache__DOT___T_34))) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
    vlSelf->Tile__DOT__icache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                                  >> 4U)));
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)) 
                 & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
                    & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
                       | (1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))))));
    vlSelf->Tile__DOT__icache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
    vlSelf->Tile__DOT__dcache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->__VdfgTmp_h92f16298__0 
                                                  >> 2U)));
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en 
        = ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)) 
           & ((IData)(vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0) 
              & (IData)(vlSelf->Tile__DOT___core_io_dcache_req_valid)));
    vlSelf->Tile__DOT__dcache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
    vlSelf->Tile__DOT__icache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en));
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable));
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                        >> 4U) : (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                  >> 4U)));
    vlSelf->Tile__DOT__dcache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en));
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable));
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                        >> 4U) : (vlSelf->__VdfgTmp_h92f16298__0 
                                  >> 2U)));
}

void VTile___024root___eval_ico(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_ico\n"); );
    // Body
    if (vlSelf->__VicoTriggered.at(0U)) {
        VTile___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void VTile___024root___eval_act(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_act\n"); );
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__0(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__0\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__icache__DOT__read_count 
        = vlSelf->Tile__DOT__icache__DOT__read_count;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0U;
    vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0U;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__4(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__4\n"); );
    // Body
    vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 = 0U;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__5(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__5\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__6(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__6\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__7(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__7\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__8(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__8\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__9(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__9\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__10(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__10\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__11(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__11\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__12(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__12\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__13(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__13\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__14(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__14\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__15(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__15\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc 
        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__16(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__16\n"); );
    // Body
    vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 = 0U;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__31(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__31\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__icache__DOT__wdata[1U]);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[1U] 
                        >> 8U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[1U] 
                        >> 0x10U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__icache__DOT__wdata[1U] 
               >> 0x18U);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__icache__DOT__wdata[2U]);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[2U] 
                        >> 8U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[2U] 
                        >> 0x10U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__icache__DOT__wdata[2U] 
               >> 0x18U);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v0))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v1))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v2))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_2_ext__DOT__Memory__v3))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v0))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v1))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v2))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_1_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__32(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__32\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0U;
    __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__icache__DOT__wdata[3U]);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[3U] 
                        >> 8U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[3U] 
                        >> 0x10U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__icache__DOT__wdata[3U] 
               >> 0x18U);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__icache__DOT__wdata[0U]);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[0U] 
                        >> 8U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__icache__DOT__wdata[0U] 
                        >> 0x10U));
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__icache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__icache__DOT__wdata[0U] 
               >> 0x18U);
        __Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__icache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v0))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v1))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v2))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_3_ext__DOT__Memory__v3))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v0))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v1))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v2))));
    }
    if (__Vdlyvset__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__icache__DOT__dataMem_0_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__33(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__33\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__dcache__DOT__wdata[1U]);
        __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[1U] 
                        >> 8U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[1U] 
                        >> 0x10U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__dcache__DOT__wdata[1U] 
               >> 0x18U);
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v0))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v1))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v2))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_1_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__34(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__34\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__dcache__DOT__wdata[2U]);
        __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[2U] 
                        >> 8U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[2U] 
                        >> 0x10U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__dcache__DOT__wdata[2U] 
               >> 0x18U);
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v0))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v1))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v2))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_2_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__35(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__35\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__dcache__DOT__wdata[3U]);
        __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[3U] 
                        >> 8U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[3U] 
                        >> 0x10U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__dcache__DOT__wdata[3U] 
               >> 0x18U);
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v0))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v1))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v2))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_3_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__36(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__36\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0;
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*4:0*/ __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    CData/*7:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3;
    __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0;
    // Body
    __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0U;
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & (IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask)) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 
            = (0xffU & vlSelf->Tile__DOT__dcache__DOT__wdata[0U]);
        __Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 = 0U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 1U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[0U] 
                        >> 8U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 = 8U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 2U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 
            = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__wdata[0U] 
                        >> 0x10U));
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 = 0x10U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if ((((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_1) 
          & ((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask) 
             >> 3U)) & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 
            = (vlSelf->Tile__DOT__dcache__DOT__wdata[0U] 
               >> 0x18U);
        vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 1U;
        __Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 = 0x18U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3 
            = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    }
    if (__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v0))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v1))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v2))));
    }
    if (vlSelf->__Vdlyvset__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3) {
        vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3] 
            = (((~ ((IData)(0xffU) << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3))) 
                & vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory
                [__Vdlyvdim0__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3]) 
               | (0xffffffffULL & ((IData)(__Vdlyvval__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3) 
                                   << (IData)(__Vdlyvlsb__Tile__DOT__dcache__DOT__dataMem_0_ext__DOT__Memory__v3))));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__37(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__37\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__arb__DOT__state = vlSelf->Tile__DOT__arb__DOT__state;
    vlSelf->__Vdly__Tile__DOT__icache__DOT__state = vlSelf->Tile__DOT__icache__DOT__state;
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 0U;
        vlSelf->__Vdly__Tile__DOT__icache__DOT__state = 0U;
    } else {
        if ((0U != (IData)(vlSelf->Tile__DOT__arb__DOT__state))) {
            if ((1U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))) {
                if ((((IData)(vlSelf->io_nasti_r_ready) 
                      & (IData)(vlSelf->io_nasti_r_valid)) 
                     & (IData)(vlSelf->io_nasti_r_bits_last))) {
                    vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 0U;
                }
            } else if ((2U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))) {
                if ((((IData)(vlSelf->io_nasti_r_ready) 
                      & (IData)(vlSelf->io_nasti_r_valid)) 
                     & (IData)(vlSelf->io_nasti_r_bits_last))) {
                    vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 0U;
                }
            } else if ((3U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))) {
                if ((((IData)(vlSelf->Tile__DOT___arb_io_dcache_w_ready) 
                      & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid)) 
                     & (IData)(vlSelf->io_nasti_w_bits_last))) {
                    vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 4U;
                }
            } else if ((((4U == (IData)(vlSelf->Tile__DOT__arb__DOT__state)) 
                         & (IData)(vlSelf->io_nasti_b_ready)) 
                        & (IData)(vlSelf->io_nasti_b_valid))) {
                vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 0U;
            }
        } else if (((IData)(vlSelf->Tile__DOT___arb_io_dcache_aw_ready) 
                    & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_aw_valid))) {
            vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 3U;
        } else if (((IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready) 
                    & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid))) {
            vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 2U;
        } else if (((IData)(vlSelf->Tile__DOT___arb_io_icache_ar_ready) 
                    & (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid))) {
            vlSelf->__Vdly__Tile__DOT__arb__DOT__state = 1U;
        }
        if ((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state))) {
            vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5 
                = (((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                    & (IData)(vlSelf->Tile__DOT__icache__DOT__read_wrap_out))
                    ? ((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask)) 
                       << 1U) : (IData)(vlSelf->Tile__DOT__icache__DOT__state));
            vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 
                = ((0xfc0000U & (((IData)(vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5) 
                                  << 0x15U) | ((IData)(vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5) 
                                               << 0x12U))) 
                   | (((((IData)(vlSelf->Tile__DOT___arb_io_icache_ar_ready) 
                         & (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid))
                         ? 6U : (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                       << 0xfU) | (((IData)(vlSelf->Tile__DOT__icache__DOT__state) 
                                    << 0xcU) | (((IData)(vlSelf->Tile__DOT__icache__DOT__state) 
                                                 << 9U) 
                                                | ((((IData)(vlSelf->Tile__DOT__icache__DOT___T_35)
                                                      ? 0U
                                                      : 
                                                     (((IData)(vlSelf->Tile__DOT___arb_io_icache_ar_ready) 
                                                       & (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid))
                                                       ? 6U
                                                       : (IData)(vlSelf->Tile__DOT__icache__DOT__state))) 
                                                    << 6U) 
                                                   | ((((IData)(vlSelf->Tile__DOT__icache__DOT__hit)
                                                         ? 
                                                        ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)
                                                          ? 0U
                                                          : 1U)
                                                         : 
                                                        (((IData)(vlSelf->Tile__DOT___arb_io_icache_ar_ready) 
                                                          & (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid))
                                                          ? 6U
                                                          : (IData)(vlSelf->Tile__DOT__icache__DOT__state))) 
                                                       << 3U) 
                                                      | (IData)(vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5)))))));
            vlSelf->__Vdly__Tile__DOT__icache__DOT__state 
                = ((0x17U >= (0x1fU & ((IData)(3U) 
                                       * (IData)(vlSelf->Tile__DOT__icache__DOT__state))))
                    ? (7U & (vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 
                             >> (0x1fU & ((IData)(3U) 
                                          * (IData)(vlSelf->Tile__DOT__icache__DOT__state)))))
                    : 0U);
        } else if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
            vlSelf->__Vdly__Tile__DOT__icache__DOT__state = 1U;
        }
    }
}

extern const VlWide<8>/*255:0*/ VTile__ConstPool__CONST_h4e9f510d_0;

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__38(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__38\n"); );
    // Init
    VlWide<8>/*255:0*/ __Vdly__Tile__DOT__icache__DOT__v;
    VL_ZERO_W(256, __Vdly__Tile__DOT__icache__DOT__v);
    VlWide<8>/*255:0*/ __Vdly__Tile__DOT__icache__DOT__d;
    VL_ZERO_W(256, __Vdly__Tile__DOT__icache__DOT__d);
    VlWide<8>/*255:0*/ __Vtemp_h7486906c__0;
    VlWide<8>/*255:0*/ __Vtemp_h46d25e4c__0;
    // Body
    __Vdly__Tile__DOT__icache__DOT__d[0U] = vlSelf->Tile__DOT__icache__DOT__d[0U];
    __Vdly__Tile__DOT__icache__DOT__d[1U] = vlSelf->Tile__DOT__icache__DOT__d[1U];
    __Vdly__Tile__DOT__icache__DOT__d[2U] = vlSelf->Tile__DOT__icache__DOT__d[2U];
    __Vdly__Tile__DOT__icache__DOT__d[3U] = vlSelf->Tile__DOT__icache__DOT__d[3U];
    __Vdly__Tile__DOT__icache__DOT__d[4U] = vlSelf->Tile__DOT__icache__DOT__d[4U];
    __Vdly__Tile__DOT__icache__DOT__d[5U] = vlSelf->Tile__DOT__icache__DOT__d[5U];
    __Vdly__Tile__DOT__icache__DOT__d[6U] = vlSelf->Tile__DOT__icache__DOT__d[6U];
    __Vdly__Tile__DOT__icache__DOT__d[7U] = vlSelf->Tile__DOT__icache__DOT__d[7U];
    __Vdly__Tile__DOT__icache__DOT__v[0U] = vlSelf->Tile__DOT__icache__DOT__v[0U];
    __Vdly__Tile__DOT__icache__DOT__v[1U] = vlSelf->Tile__DOT__icache__DOT__v[1U];
    __Vdly__Tile__DOT__icache__DOT__v[2U] = vlSelf->Tile__DOT__icache__DOT__v[2U];
    __Vdly__Tile__DOT__icache__DOT__v[3U] = vlSelf->Tile__DOT__icache__DOT__v[3U];
    __Vdly__Tile__DOT__icache__DOT__v[4U] = vlSelf->Tile__DOT__icache__DOT__v[4U];
    __Vdly__Tile__DOT__icache__DOT__v[5U] = vlSelf->Tile__DOT__icache__DOT__v[5U];
    __Vdly__Tile__DOT__icache__DOT__v[6U] = vlSelf->Tile__DOT__icache__DOT__v[6U];
    __Vdly__Tile__DOT__icache__DOT__v[7U] = vlSelf->Tile__DOT__icache__DOT__v[7U];
    if (vlSelf->reset) {
        __Vdly__Tile__DOT__icache__DOT__v[0U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[1U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[2U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[3U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[4U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[5U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[6U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__v[7U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[0U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[1U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[2U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[3U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[4U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[5U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[6U] = 0U;
        __Vdly__Tile__DOT__icache__DOT__d[7U] = 0U;
    } else {
        __Vtemp_h7486906c__0[0U] = (0xffU & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                             >> 4U));
        __Vtemp_h7486906c__0[1U] = 0U;
        __Vtemp_h7486906c__0[2U] = 0U;
        __Vtemp_h7486906c__0[3U] = 0U;
        __Vtemp_h7486906c__0[4U] = 0U;
        __Vtemp_h7486906c__0[5U] = 0U;
        __Vtemp_h7486906c__0[6U] = 0U;
        __Vtemp_h7486906c__0[7U] = 0U;
        VL_SHIFTL_WWW(256,256,256, __Vtemp_h46d25e4c__0, VTile__ConstPool__CONST_h4e9f510d_0, __Vtemp_h7486906c__0);
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[0U] 
            = __Vtemp_h46d25e4c__0[0U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[1U] 
            = __Vtemp_h46d25e4c__0[1U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[2U] 
            = __Vtemp_h46d25e4c__0[2U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[3U] 
            = __Vtemp_h46d25e4c__0[3U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[4U] 
            = __Vtemp_h46d25e4c__0[4U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[5U] 
            = __Vtemp_h46d25e4c__0[5U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[6U] 
            = __Vtemp_h46d25e4c__0[6U];
        vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[7U] 
            = __Vtemp_h46d25e4c__0[7U];
        __Vdly__Tile__DOT__icache__DOT__v[0U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[0U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[0U]);
        __Vdly__Tile__DOT__icache__DOT__v[1U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[1U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[1U]);
        __Vdly__Tile__DOT__icache__DOT__v[2U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[2U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[2U]);
        __Vdly__Tile__DOT__icache__DOT__v[3U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[3U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[3U]);
        __Vdly__Tile__DOT__icache__DOT__v[4U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[4U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[4U]);
        __Vdly__Tile__DOT__icache__DOT__v[5U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[5U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[5U]);
        __Vdly__Tile__DOT__icache__DOT__v[6U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[6U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[6U]);
        __Vdly__Tile__DOT__icache__DOT__v[7U] = (((- (IData)((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) 
                                                  & vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[7U]) 
                                                 | vlSelf->Tile__DOT__icache__DOT__v[7U]);
        if (vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en) {
            if (vlSelf->Tile__DOT__icache__DOT__is_alloc) {
                __Vdly__Tile__DOT__icache__DOT__d[0U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[0U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[0U]));
                __Vdly__Tile__DOT__icache__DOT__d[1U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[1U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[1U]));
                __Vdly__Tile__DOT__icache__DOT__d[2U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[2U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[2U]));
                __Vdly__Tile__DOT__icache__DOT__d[3U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[3U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[3U]));
                __Vdly__Tile__DOT__icache__DOT__d[4U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[4U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[4U]));
                __Vdly__Tile__DOT__icache__DOT__d[5U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[5U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[5U]));
                __Vdly__Tile__DOT__icache__DOT__d[6U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[6U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[6U]));
                __Vdly__Tile__DOT__icache__DOT__d[7U] 
                    = (~ ((~ vlSelf->Tile__DOT__icache__DOT__d[7U]) 
                          | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[7U]));
            } else {
                __Vdly__Tile__DOT__icache__DOT__d[0U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[0U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[0U]);
                __Vdly__Tile__DOT__icache__DOT__d[1U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[1U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[1U]);
                __Vdly__Tile__DOT__icache__DOT__d[2U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[2U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[2U]);
                __Vdly__Tile__DOT__icache__DOT__d[3U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[3U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[3U]);
                __Vdly__Tile__DOT__icache__DOT__d[4U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[4U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[4U]);
                __Vdly__Tile__DOT__icache__DOT__d[5U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[5U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[5U]);
                __Vdly__Tile__DOT__icache__DOT__d[6U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[6U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[6U]);
                __Vdly__Tile__DOT__icache__DOT__d[7U] 
                    = (vlSelf->Tile__DOT__icache__DOT__d[7U] 
                       | vlSelf->Tile__DOT__icache__DOT__unnamedblk1__DOT___d_T_1[7U]);
            }
        }
    }
    vlSelf->Tile__DOT__icache__DOT__d[0U] = __Vdly__Tile__DOT__icache__DOT__d[0U];
    vlSelf->Tile__DOT__icache__DOT__d[1U] = __Vdly__Tile__DOT__icache__DOT__d[1U];
    vlSelf->Tile__DOT__icache__DOT__d[2U] = __Vdly__Tile__DOT__icache__DOT__d[2U];
    vlSelf->Tile__DOT__icache__DOT__d[3U] = __Vdly__Tile__DOT__icache__DOT__d[3U];
    vlSelf->Tile__DOT__icache__DOT__d[4U] = __Vdly__Tile__DOT__icache__DOT__d[4U];
    vlSelf->Tile__DOT__icache__DOT__d[5U] = __Vdly__Tile__DOT__icache__DOT__d[5U];
    vlSelf->Tile__DOT__icache__DOT__d[6U] = __Vdly__Tile__DOT__icache__DOT__d[6U];
    vlSelf->Tile__DOT__icache__DOT__d[7U] = __Vdly__Tile__DOT__icache__DOT__d[7U];
    vlSelf->Tile__DOT__icache__DOT__v[0U] = __Vdly__Tile__DOT__icache__DOT__v[0U];
    vlSelf->Tile__DOT__icache__DOT__v[1U] = __Vdly__Tile__DOT__icache__DOT__v[1U];
    vlSelf->Tile__DOT__icache__DOT__v[2U] = __Vdly__Tile__DOT__icache__DOT__v[2U];
    vlSelf->Tile__DOT__icache__DOT__v[3U] = __Vdly__Tile__DOT__icache__DOT__v[3U];
    vlSelf->Tile__DOT__icache__DOT__v[4U] = __Vdly__Tile__DOT__icache__DOT__v[4U];
    vlSelf->Tile__DOT__icache__DOT__v[5U] = __Vdly__Tile__DOT__icache__DOT__v[5U];
    vlSelf->Tile__DOT__icache__DOT__v[6U] = __Vdly__Tile__DOT__icache__DOT__v[6U];
    vlSelf->Tile__DOT__icache__DOT__v[7U] = __Vdly__Tile__DOT__icache__DOT__v[7U];
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__39(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__39\n"); );
    // Body
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__state = vlSelf->Tile__DOT__dcache__DOT__state;
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__state = 0U;
    } else if ((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state))) {
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5 
            = (((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_wrap_out))
                ? ((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask)) 
                   << 1U) : (IData)(vlSelf->Tile__DOT__dcache__DOT__state));
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 
            = ((0xfc0000U & (((IData)(vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5) 
                              << 0x15U) | ((IData)(vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5) 
                                           << 0x12U))) 
               | (((((IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready) 
                     & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid))
                     ? 6U : (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                   << 0xfU) | (((((IData)(vlSelf->Tile__DOT___dcache_io_nasti_b_ready) 
                                  & ((IData)(vlSelf->io_nasti_b_valid) 
                                     & (4U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))))
                                  ? 5U : (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                << 0xcU) | ((((IData)(vlSelf->io_nasti_w_bits_last)
                                               ? 4U
                                               : (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                             << 9U) 
                                            | ((((IData)(vlSelf->Tile__DOT__dcache__DOT___T_35)
                                                  ? 0U
                                                  : 
                                                 (((IData)(vlSelf->Tile__DOT___arb_io_dcache_aw_ready) 
                                                   & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_aw_valid))
                                                   ? 3U
                                                   : 
                                                  (((IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready) 
                                                    & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid))
                                                    ? 6U
                                                    : (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                                                << 6U) 
                                               | ((((IData)(vlSelf->Tile__DOT__dcache__DOT__hit)
                                                     ? 
                                                    ((IData)(vlSelf->Tile__DOT___core_io_dcache_req_valid)
                                                      ? 
                                                     ((0U 
                                                       != (IData)(vlSelf->Tile__DOT___core_io_dcache_req_bits_mask))
                                                       ? 2U
                                                       : 1U)
                                                      : 0U)
                                                     : 
                                                    (((IData)(vlSelf->Tile__DOT___arb_io_dcache_aw_ready) 
                                                      & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_aw_valid))
                                                      ? 3U
                                                      : 
                                                     (((IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready) 
                                                       & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid))
                                                       ? 6U
                                                       : (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                                                   << 3U) 
                                                  | (IData)(vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_5)))))));
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__state 
            = ((0x17U >= (0x1fU & ((IData)(3U) * (IData)(vlSelf->Tile__DOT__dcache__DOT__state))))
                ? (7U & (vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT__unnamedblk2__DOT___GEN_6 
                         >> (0x1fU & ((IData)(3U) * (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))))
                : 0U);
    } else if (vlSelf->Tile__DOT___core_io_dcache_req_valid) {
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__state 
            = ((0U != (IData)(vlSelf->Tile__DOT___core_io_dcache_req_bits_mask))
                ? 2U : 1U);
    }
    if (vlSelf->Tile__DOT___dcache_io_cpu_resp_valid) {
        vlSelf->Tile__DOT__dcache__DOT__cpu_data = 
            (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
             << (0x18U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
                          << 3U)));
        vlSelf->Tile__DOT__dcache__DOT__cpu_mask = vlSelf->Tile__DOT___core_io_dcache_req_bits_mask;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__40(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__40\n"); );
    // Init
    VlWide<8>/*255:0*/ __Vtemp_h1e18caf7__0;
    VlWide<8>/*255:0*/ __Vtemp_hb5646b87__0;
    // Body
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[0U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[0U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[1U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[1U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[2U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[2U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[3U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[3U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[4U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[4U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[5U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[5U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[6U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[6U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[7U] = 
        vlSelf->Tile__DOT__dcache__DOT__d[7U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[0U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[0U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[1U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[1U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[2U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[2U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[3U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[3U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[4U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[4U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[5U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[5U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[6U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[6U];
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[7U] = 
        vlSelf->Tile__DOT__dcache__DOT__v[7U];
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[0U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[1U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[2U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[3U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[4U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[5U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[6U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[7U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[0U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[1U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[2U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[3U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[4U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[5U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[6U] = 0U;
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[7U] = 0U;
    } else {
        __Vtemp_h1e18caf7__0[0U] = (0xffU & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                             >> 4U));
        __Vtemp_h1e18caf7__0[1U] = 0U;
        __Vtemp_h1e18caf7__0[2U] = 0U;
        __Vtemp_h1e18caf7__0[3U] = 0U;
        __Vtemp_h1e18caf7__0[4U] = 0U;
        __Vtemp_h1e18caf7__0[5U] = 0U;
        __Vtemp_h1e18caf7__0[6U] = 0U;
        __Vtemp_h1e18caf7__0[7U] = 0U;
        VL_SHIFTL_WWW(256,256,256, __Vtemp_hb5646b87__0, VTile__ConstPool__CONST_h4e9f510d_0, __Vtemp_h1e18caf7__0);
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[0U] 
            = __Vtemp_hb5646b87__0[0U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[1U] 
            = __Vtemp_hb5646b87__0[1U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[2U] 
            = __Vtemp_hb5646b87__0[2U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[3U] 
            = __Vtemp_hb5646b87__0[3U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[4U] 
            = __Vtemp_hb5646b87__0[4U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[5U] 
            = __Vtemp_hb5646b87__0[5U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[6U] 
            = __Vtemp_hb5646b87__0[6U];
        vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[7U] 
            = __Vtemp_hb5646b87__0[7U];
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[0U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[0U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[0U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[1U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[1U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[1U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[2U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[2U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[2U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[3U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[3U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[3U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[4U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[4U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[4U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[5U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[5U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[5U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[6U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[6U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[6U]);
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[7U] 
            = (((- (IData)((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) 
                & vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[7U]) 
               | vlSelf->Tile__DOT__dcache__DOT__v[7U]);
        if (vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en) {
            if (vlSelf->Tile__DOT__dcache__DOT__is_alloc) {
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[0U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[0U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[0U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[1U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[1U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[1U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[2U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[2U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[2U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[3U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[3U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[3U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[4U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[4U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[4U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[5U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[5U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[5U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[6U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[6U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[6U]));
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[7U] 
                    = (~ ((~ vlSelf->Tile__DOT__dcache__DOT__d[7U]) 
                          | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[7U]));
            } else {
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[0U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[0U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[0U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[1U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[1U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[1U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[2U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[2U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[2U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[3U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[3U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[3U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[4U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[4U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[4U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[5U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[5U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[5U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[6U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[6U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[6U]);
                vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[7U] 
                    = (vlSelf->Tile__DOT__dcache__DOT__d[7U] 
                       | vlSelf->Tile__DOT__dcache__DOT__unnamedblk1__DOT___d_T_1[7U]);
            }
        }
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__41(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__41\n"); );
    // Body
    if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT___T_1) {
        vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 
            = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask)
                : 0U);
        vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 = 1U;
        vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0 
            = (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                        >> 7U));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__42(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__42\n"); );
    // Body
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc = 0x1fcULL;
    } else if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
        if (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc 
                = (QData)((IData)(((IData)(0x100U) 
                                   + ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV) 
                                      << 6U))));
        } else if ((3U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel))) {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc 
                = (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc));
        } else if (vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_4) {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc 
                = ((QData)((IData)((vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
                                    >> 1U))) << 1U);
        } else if ((2U != (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel))) {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_1;
        }
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__43(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__43\n"); );
    // Init
    VlWide<9>/*287:0*/ __Vtemp_he6e910e3__0;
    // Body
    if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT___T_1) {
        __Vtemp_he6e910e3__0[0U] = (IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)));
        __Vtemp_he6e910e3__0[1U] = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__bypass__io_idata_load 
                                     << 1U) | (IData)(
                                                      ((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)) 
                                                       >> 0x20U)));
        __Vtemp_he6e910e3__0[2U] = ((((IData)(4U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc) 
                                     << 2U) | ((2U 
                                                & (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__bypass__io_idata_load 
                                                   >> 0x1eU)) 
                                               | (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__bypass__io_idata_load 
                                                  >> 0x1fU)));
        __Vtemp_he6e910e3__0[3U] = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out 
                                     << 3U) | (((IData)(4U) 
                                                + vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc) 
                                               >> 0x1eU));
        __Vtemp_he6e910e3__0[4U] = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                                     << 4U) | (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out 
                                               >> 0x1dU));
        __Vtemp_he6e910e3__0[5U] = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul 
                                     << 5U) | (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                                               >> 0x1cU));
        __Vtemp_he6e910e3__0[6U] = (((IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu))) 
                                     << 6U) | (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul 
                                               >> 0x1bU));
        __Vtemp_he6e910e3__0[7U] = (((IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu))) 
                                     >> 0x1aU) | (((IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu))) 
                                                   << 7U) 
                                                  | ((IData)(
                                                             ((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)) 
                                                              >> 0x20U)) 
                                                     << 6U)));
        __Vtemp_he6e910e3__0[8U] = (((0x3fU & ((IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu))) 
                                               >> 0x19U)) 
                                     | ((IData)(((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)) 
                                                 >> 0x20U)) 
                                        >> 0x1aU)) 
                                    | (0xc0U & ((0x40U 
                                                 & ((IData)((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu))) 
                                                    >> 0x19U)) 
                                                | ((IData)(
                                                           ((QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)) 
                                                            >> 0x20U)) 
                                                   << 7U))));
        vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 
            = ((0x107U >= (0x1ffU & ((IData)(0x21U) 
                                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))))
                ? (((0U == (0x1fU & ((IData)(0x21U) 
                                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))))
                     ? 0U : (__Vtemp_he6e910e3__0[(
                                                   ((IData)(0x1fU) 
                                                    + 
                                                    (0x1ffU 
                                                     & ((IData)(0x21U) 
                                                        * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)))) 
                                                   >> 5U)] 
                             << ((IData)(0x20U) - (0x1fU 
                                                   & ((IData)(0x21U) 
                                                      * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)))))) 
                   | (__Vtemp_he6e910e3__0[(0xfU & 
                                            (((IData)(0x21U) 
                                              * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)) 
                                             >> 5U))] 
                      >> (0x1fU & ((IData)(0x21U) * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)))))
                : 0U);
        vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 = 1U;
        vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0 
            = (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                        >> 7U));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__44(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__44\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN_0 
        = vlSelf->Tile__DOT__icache__DOT___GEN_1;
    vlSelf->Tile__DOT__icache__DOT__dataMem_0_ext__DOT___GEN 
        = vlSelf->Tile__DOT__icache__DOT___GEN_0;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__45(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__45\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    IData/*19:0*/ __Vdlyvval__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    // Body
    vlSelf->__Vdly__Tile__DOT__dcache__DOT__read_count 
        = vlSelf->Tile__DOT__dcache__DOT__read_count;
    __Vdlyvset__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0U;
    if (((IData)(vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 
            = (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
               >> 0xcU);
        __Vdlyvset__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 1U;
        __Vdlyvdim0__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr;
    }
    vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN_0 
        = vlSelf->Tile__DOT__dcache__DOT___GEN_1;
    vlSelf->Tile__DOT__dcache__DOT__dataMem_0_ext__DOT___GEN 
        = vlSelf->Tile__DOT__dcache__DOT___GEN_0;
    if (vlSelf->Tile__DOT__dcache__DOT__ren_reg) {
        vlSelf->Tile__DOT__dcache__DOT__rdata_buf[0U] 
            = vlSelf->Tile__DOT__dcache__DOT__rdata[0U];
        vlSelf->Tile__DOT__dcache__DOT__rdata_buf[1U] 
            = vlSelf->Tile__DOT__dcache__DOT__rdata[1U];
        vlSelf->Tile__DOT__dcache__DOT__rdata_buf[2U] 
            = vlSelf->Tile__DOT__dcache__DOT__rdata[2U];
        vlSelf->Tile__DOT__dcache__DOT__rdata_buf[3U] 
            = vlSelf->Tile__DOT__dcache__DOT__rdata[3U];
    }
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_0 
        = vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en;
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN 
        = vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr;
    vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg = vlSelf->Tile__DOT__dcache__DOT__is_alloc;
    vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1 
        = vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en;
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__read_count = 0U;
    } else if (vlSelf->Tile__DOT__dcache__DOT___T_22) {
        vlSelf->__Vdly__Tile__DOT__dcache__DOT__read_count 
            = (1U & ((IData)(vlSelf->Tile__DOT__dcache__DOT__read_count) 
                     - (IData)(1U)));
    }
    if (((IData)(vlSelf->Tile__DOT__dcache__DOT___T_22) 
         & (~ (IData)(vlSelf->Tile__DOT__dcache__DOT__read_count)))) {
        vlSelf->Tile__DOT__dcache__DOT__refill_buf_0 
            = vlSelf->io_nasti_r_bits_data;
    }
    if (((IData)(vlSelf->Tile__DOT__dcache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_count))) {
        vlSelf->Tile__DOT__dcache__DOT__refill_buf_1 
            = vlSelf->io_nasti_r_bits_data;
    }
    if (__Vdlyvset__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0] 
            = __Vdlyvval__Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    }
    vlSelf->Tile__DOT__dcache__DOT__ren_reg = vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en;
    vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
        = (((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
            & (IData)(vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN_0))
            ? vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT__Memory
           [vlSelf->Tile__DOT__dcache__DOT__metaMem_tag_ext__DOT___GEN]
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__46(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__46\n"); );
    // Init
    CData/*7:0*/ __Vdlyvdim0__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvdim0__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    IData/*19:0*/ __Vdlyvval__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvval__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    CData/*0:0*/ __Vdlyvset__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    __Vdlyvset__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0;
    // Body
    __Vdlyvset__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 0U;
    if (((IData)(vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en))) {
        __Vdlyvval__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 
            = (vlSelf->Tile__DOT__icache__DOT__addr_reg 
               >> 0xcU);
        __Vdlyvset__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 = 1U;
        __Vdlyvdim0__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0 
            = vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr;
    }
    if (vlSelf->Tile__DOT___icache_io_cpu_resp_valid) {
        vlSelf->Tile__DOT__icache__DOT__cpu_data = 0U;
        vlSelf->Tile__DOT__icache__DOT__cpu_mask = 0U;
    }
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_0 
        = vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en;
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN 
        = vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr;
    vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1 
        = vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en;
    if (__Vdlyvset__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory[__Vdlyvdim0__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0] 
            = __Vdlyvval__Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory__v0;
    }
    vlSelf->Tile__DOT__arb__DOT__state = vlSelf->__Vdly__Tile__DOT__arb__DOT__state;
    vlSelf->Tile__DOT__icache__DOT__state = vlSelf->__Vdly__Tile__DOT__icache__DOT__state;
    vlSelf->Tile__DOT__dcache__DOT__state = vlSelf->__Vdly__Tile__DOT__dcache__DOT__state;
    vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0 
        = (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                 | (1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))));
    vlSelf->Tile__DOT__dcache__DOT___GEN_4 = ((IData)(vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0) 
                                              | (2U 
                                                 == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__47(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__47\n"); );
    // Body
    if (vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory[vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0] 
            = vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory__v0;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__48(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__48\n"); );
    // Body
    if (vlSelf->__Vdlyvset__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory[vlSelf->__Vdlyvdim0__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0] 
            = vlSelf->__Vdlyvval__Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory__v0;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__54(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__54\n"); );
    // Body
    if (vlSelf->Tile__DOT___dcache_io_cpu_resp_valid) {
        vlSelf->Tile__DOT__dcache__DOT__addr_reg = 
            (vlSelf->__VdfgTmp_h92f16298__0 << 2U);
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__55(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__55\n"); );
    // Init
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret = 0;
    IData/*31:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T = 0;
    IData/*31:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7 = 0;
    IData/*31:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18;
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18 = 0;
    // Body
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret 
        = (IData)(((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
                   & (0x10000000U == (0x10100000U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst))));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out 
           | vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in);
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_out 
           & (~ vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5 
        = (2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7 
        = (3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
        = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
            ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
            : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                    : 0U)));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7 
        = (0x300U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8 
        = (0x344U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) 
           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9 
        = (0x304U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10 
        = (0x701U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11 
        = (0x741U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12 
        = (0x321U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13 
        = (0x340U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14 
        = (0x341U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15 
        = (0x342U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16 
        = (0x343U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17 
        = (0x780U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18 
        = (0x781U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                      >> 0x14U));
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV = 3U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 = 3U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE = 0U;
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIP = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIE = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIP = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIE = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qstate = 0U;
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19 
            = (0x900U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20 
            = (0x901U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21 
            = (0x902U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22 
            = (0x980U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23 
            = (0x981U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_24 
            = (0x982U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_25 
            = (0x721U == (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                          >> 0x14U));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__isInstRet 
            = (((0x13U != vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst) 
                & (((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall)) 
                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak))) 
               & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)));
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 
            = ((1U & (((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                          | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                      | (~ ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10) 
                            | (~ ((((((((((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11) 
                                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                                  | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20))))))))
                ? ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0)
                : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U))));
        if ((1U & ((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                        | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                   | (~ ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11) 
                         | (~ ((((((((((((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12) 
                                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                                 | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22)) 
                               | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23))))))))) {
            if ((0xffffffffU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0)) {
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh 
                    = ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh);
            }
        } else {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle 
            = ((1U & ((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                      | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19))))
                ? ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle)
                : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U))));
        if ((1U & (((((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                      | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                     | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21)) 
                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22))))) {
            if ((0xffffffffU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle)) {
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh 
                    = ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh);
            }
        } else {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        if ((1U & ((((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                  | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                     | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21))))) {
            if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__isInstRet) {
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret 
                    = ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret);
            }
        } else {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        if ((1U & (((((((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                     | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                        | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                       | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                      | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21)) 
                     | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23)) 
                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_24))))) {
            if (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__isInstRet) 
                 & (0xffffffffU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret))) {
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth 
                    = ((IData)(1U) + vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth);
            }
        } else {
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__unnamedblk3__DOT___GEN_0 
                = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen) 
                   & (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7));
            if (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) {
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 
                    = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV;
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 
                    = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE;
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV = 3U;
            } else {
                if (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret) {
                    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV 
                        = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1;
                    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 = 0U;
                } else if (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__unnamedblk3__DOT___GEN_0) {
                    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV 
                        = (3U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                                 >> 1U));
                    vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 
                        = (3U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                                 >> 4U));
                }
                vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 
                    = (1U & ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret) 
                             | ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__unnamedblk3__DOT___GEN_0)
                                 ? (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                                    >> 3U) : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1))));
            }
            vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE 
                = (1U & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)) 
                         & ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret)
                             ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1)
                             : ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT__unnamedblk3__DOT___GEN_0)
                                 ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata
                                 : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE)))));
        }
        if ((1U & (~ (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                        | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                      | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)))))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIP 
                = (1U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                         >> 7U));
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIP 
                = (1U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                         >> 3U));
        }
        if ((1U & (~ ((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                         | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                      | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)))))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MTIE 
                = (1U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                         >> 7U));
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__MSIE 
                = (1U & (Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata 
                         >> 3U));
        }
        if ((1U & (~ ((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                 | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                      | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)))))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        if ((1U & (~ ((((((((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                         | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                            | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                          | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21)) 
                         | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22)) 
                        | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23)) 
                       | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_24)) 
                      | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_25)))))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
        if ((1U & (~ (((((((((((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                                          | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18)) 
                             | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_19)) 
                            | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_20)) 
                           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_21)) 
                          | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_22)) 
                         | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_23)) 
                        | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_24)) 
                       | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__unnamedblk2__DOT___T_25)) 
                      | (0x722U != (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                    >> 0x14U)))))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qstate 
                = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                    : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                        : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                            : 0U)));
        }
    }
    if ((1U & (~ (((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                        | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                  | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)))))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtimecmp 
            = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                    : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                        : 0U)));
    }
    if ((1U & (~ ((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                         | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                  | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)))))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mscratch 
            = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                    : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                        : 0U)));
    }
    if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
        if (vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc 
                = (0xfffffffcU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc);
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mcause 
                = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid)
                    ? 0U : ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid)
                             ? 4U : ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid)
                                      ? 6U : (0xfU 
                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall)
                                                  ? 
                                                 ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV) 
                                                  - (IData)(8U))
                                                  : 
                                                 (2U 
                                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak)))))));
            if ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid) 
                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid)) 
                 | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid))) {
                vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mbadaddr 
                    = vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu;
            }
        } else {
            if ((1U & (~ ((((((((((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret) 
                                  | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                          | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)))))) {
                vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mepc 
                    = (0xfffffffcU & Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata);
            }
            if ((1U & (~ (((((((((((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret) 
                                   | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                          | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)))))) {
                vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mcause 
                    = (0x8000000fU & Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__wdata);
            }
            if ((1U & (~ ((((((((((((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT__isEret) 
                                    | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                               | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                              | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                             | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                           | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                          | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)))))) {
                vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mbadaddr 
                    = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                        ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                        : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                            ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                            : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                                : 0U)));
            }
        }
    }
    if ((1U & (((((((((((((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall) 
                            | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___GEN)) 
                           | (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen))) 
                          | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_7)) 
                         | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_8)) 
                        | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_9)) 
                       | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_10)) 
                      | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_11)) 
                     | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_12)) 
                    | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_13)) 
                   | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_14)) 
                  | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_15)) 
                 | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_16)) 
                | (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_17)) 
               | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___T_18))))) {
        if (vlSelf->io_host_fromhost_valid) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mfromhost 
                = vlSelf->io_host_fromhost_bits;
        }
    } else {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mfromhost 
            = ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_7)
                ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_2
                : ((IData)(Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T_5)
                    ? Tile__DOT__core__DOT__dpath__DOT__csr__DOT__unnamedblk1__DOT___wdata_T
                    : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd))
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in
                        : 0U)));
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__66(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__66\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE1;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__67(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__67\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__IE;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__68(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__68\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV1;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__69(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__69\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instreth;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__70(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__70\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__instret;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__71(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__71\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycleh;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__72(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__72\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__cycle;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__73(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__73\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__timeh;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__74(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__74\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__time_0;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__77(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__77\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV 
        = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__84(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__84\n"); );
    // Body
    vlSelf->io_host_tohost = vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__mtohost;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__85(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__85\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_17 
        = ((0x7fffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                             >> 0x1dU)))) 
                           << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                               >> 0x16U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__86(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__86\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_19 
        = (((QData)((IData)((0x7fffffffU & (- (IData)(
                                                      (1U 
                                                       & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                          >> 0x1dU))))))) 
            << 8U) | (QData)((IData)((0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                               >> 0x16U)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__87(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__87\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19 
        = ((0x3ffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x1dU)))) 
                          << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                              >> 0x16U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__88(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__88\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21 
        = ((0x3ffff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x10U)))) 
                          << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                              >> 9U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__89(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__89\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0 
        = ((0x3ff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                          >> 0x1dU)))) 
                        << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x16U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__90(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__90\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2 
        = ((0x3ff00U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                          >> 0x10U)))) 
                        << 8U)) | (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 9U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__91(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__91\n"); );
    // Init
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4;
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7;
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7 = 0;
    CData/*0:0*/ Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___GEN;
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___GEN = 0;
    // Body
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4 
        = ((IData)(vlSelf->reset) | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
                                     & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)));
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
                 & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt))));
    Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___GEN 
        = (1U & ((IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4) 
                 | (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7))));
    if (vlSelf->reset) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst = 0x13U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul = 0U;
    } else if ((1U & (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___GEN)))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pc 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_pc;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
            = vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_out;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_csr_in 
            = ((6U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_imm_sel))
                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___immGen_io_out
                : ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild)
                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out
                    : vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1));
        if ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mm8x8_io_qvm_out;
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                = (((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3) 
                                      >> 8U)))) << 9U) 
                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3));
        } else if ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask 
                = ((0U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
                    ? ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                        ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in)
                        : 0U) : 0U);
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mul8x8_io_out;
        } else if ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___br8x8_io_qvm_out;
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
                    : 0U);
        } else if ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))) {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask = 0U;
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___alu8x8_io_out;
        } else {
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in;
            vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe 
                = vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1;
        }
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul 
            = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h5a0d4b50__0)
                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
                : (((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                    | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                       | (1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out))))
                    ? (IData)((vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT___GEN_0 
                               >> 0x20U)) : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out))
                                              ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT___GEN_0)
                                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)));
    }
    if (Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type = 0U;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd = 0U;
    } else if (Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type 
            = vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type;
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type 
            = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0)
                ? 0U : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                         ? 3U : ((0x83U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                  ? 2U : ((0x103U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                           ? 1U : (
                                                   (0x203U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                    ? 5U
                                                    : 
                                                   ((0x283U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                    << 2U))))));
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd 
            = (((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_24))
                ? 0U : ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                         ? 1U : ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                  ? 2U : ((0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                           ? 3U : (
                                                   (0x2f3U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                    ? 1U
                                                    : 
                                                   ((0x373U 
                                                     == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                     ? 2U
                                                     : 
                                                    ((0x3f3U 
                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                                                      ? 3U
                                                      : 
                                                     ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23) 
                                                      << 2U))))))));
    }
    if ((1U & (~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___GEN)))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel 
            = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)
                ? 0U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9)
                         ? 2U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)
                                  ? 0U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7)
                                           ? 1U : ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_24)
                                                    ? 0U
                                                    : 
                                                   (((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14) 
                                                     | ((0x2f3U 
                                                         == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                        | ((0x373U 
                                                            == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                           | ((0x3f3U 
                                                               == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                                                              | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23)))))
                                                     ? 3U
                                                     : 
                                                    ((0x10200073U 
                                                      == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)
                                                      ? 0U
                                                      : 
                                                     ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22)
                                                       ? 5U
                                                       : 
                                                      ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h990c992e__0) 
                                                       << 2U)))))))));
    }
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en 
        = ((~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4)) 
           & ((IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7)
               ? ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_21) 
                  | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7) 
                        | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20)) 
                           & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_17) 
                              | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15)) 
                                 & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4) 
                                    | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12)) 
                                       & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0)))))))))
               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en)));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__illegal 
        = (1U & ((~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4)) 
                 & ((IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7)
                     ? (~ ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hdfe1f980__0) 
                           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0)))
                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__illegal))));
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc_check 
        = ((~ (IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_4)) 
           & ((IData)(Tile__DOT__core__DOT__dpath__DOT__unnamedblk1__DOT___T_7)
               ? (1U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel))
               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc_check)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__94(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__94\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT__read_count = vlSelf->__Vdly__Tile__DOT__dcache__DOT__read_count;
    vlSelf->io_nasti_r_ready = (((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                 & (1U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))) 
                                | ((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                   & (2U == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT__dcache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (2U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
    vlSelf->Tile__DOT__dcache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_count));
    vlSelf->io_nasti_aw_bits_addr = ((vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
                                      << 0xcU) | (0xff0U 
                                                  & vlSelf->Tile__DOT__dcache__DOT__addr_reg));
    vlSelf->Tile__DOT___dcache_io_nasti_b_ready = (
                                                   (~ 
                                                    ((IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_4) 
                                                     | (3U 
                                                        == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
                                                   & (4U 
                                                      == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
    vlSelf->Tile__DOT__dcache__DOT___wmask_T_2 = (0x7ffffU 
                                                  & ((IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask) 
                                                     << 
                                                     (0xcU 
                                                      & vlSelf->Tile__DOT__dcache__DOT__addr_reg)));
    vlSelf->Tile__DOT__dcache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__dcache__DOT__read_wrap_out));
    vlSelf->io_nasti_b_ready = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_b_ready) 
                                & (4U == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__109(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__109\n"); );
    // Init
    VlWide<4>/*127:0*/ Tile__DOT__dcache__DOT___GEN_3;
    VL_ZERO_W(128, Tile__DOT__dcache__DOT___GEN_3);
    CData/*0:0*/ __Vdly__Tile__DOT__dcache__DOT__write_count;
    __Vdly__Tile__DOT__dcache__DOT__write_count = 0;
    // Body
    __Vdly__Tile__DOT__dcache__DOT__write_count = vlSelf->Tile__DOT__dcache__DOT__write_count;
    if (vlSelf->reset) {
        __Vdly__Tile__DOT__dcache__DOT__write_count = 0U;
    } else if (vlSelf->Tile__DOT__dcache__DOT___T_1) {
        __Vdly__Tile__DOT__dcache__DOT__write_count 
            = (1U & ((IData)(vlSelf->Tile__DOT__dcache__DOT__write_count) 
                     - (IData)(1U)));
    }
    vlSelf->Tile__DOT__dcache__DOT__write_count = __Vdly__Tile__DOT__dcache__DOT__write_count;
    vlSelf->Tile__DOT___arb_io_dcache_aw_ready = ((~ (IData)(
                                                             (0U 
                                                              != (IData)(vlSelf->Tile__DOT__arb__DOT__state)))) 
                                                  & (IData)(vlSelf->io_nasti_aw_ready));
    vlSelf->Tile__DOT___arb_io_dcache_w_ready = ((IData)(vlSelf->io_nasti_w_ready) 
                                                 & (3U 
                                                    == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
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
    vlSelf->Tile__DOT___dcache_io_nasti_w_valid = (
                                                   (~ (IData)(vlSelf->Tile__DOT__dcache__DOT___GEN_4)) 
                                                   & (3U 
                                                      == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)));
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
    vlSelf->io_nasti_w_valid = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid) 
                                & (3U == (IData)(vlSelf->Tile__DOT__arb__DOT__state)));
    vlSelf->Tile__DOT__dcache__DOT___T_1 = ((IData)(vlSelf->Tile__DOT___arb_io_dcache_w_ready) 
                                            & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_w_valid));
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
    vlSelf->io_nasti_w_bits_last = ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_1) 
                                    & (IData)(vlSelf->Tile__DOT__dcache__DOT__write_count));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__110(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__110\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__111(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__111\n"); );
    // Init
    CData/*0:0*/ Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0;
    Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0 = 0;
    VlWide<8>/*255:0*/ __Vtemp_h5336a7a2__0;
    VlWide<8>/*255:0*/ __Vtemp_hc380ab92__0;
    // Body
    vlSelf->Tile__DOT__dcache__DOT__d[0U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[0U];
    vlSelf->Tile__DOT__dcache__DOT__d[1U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[1U];
    vlSelf->Tile__DOT__dcache__DOT__d[2U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[2U];
    vlSelf->Tile__DOT__dcache__DOT__d[3U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[3U];
    vlSelf->Tile__DOT__dcache__DOT__d[4U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[4U];
    vlSelf->Tile__DOT__dcache__DOT__d[5U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[5U];
    vlSelf->Tile__DOT__dcache__DOT__d[6U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[6U];
    vlSelf->Tile__DOT__dcache__DOT__d[7U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__d[7U];
    vlSelf->Tile__DOT__dcache__DOT__v[0U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[0U];
    vlSelf->Tile__DOT__dcache__DOT__v[1U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[1U];
    vlSelf->Tile__DOT__dcache__DOT__v[2U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[2U];
    vlSelf->Tile__DOT__dcache__DOT__v[3U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[3U];
    vlSelf->Tile__DOT__dcache__DOT__v[4U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[4U];
    vlSelf->Tile__DOT__dcache__DOT__v[5U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[5U];
    vlSelf->Tile__DOT__dcache__DOT__v[6U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[6U];
    vlSelf->Tile__DOT__dcache__DOT__v[7U] = vlSelf->__Vdly__Tile__DOT__dcache__DOT__v[7U];
    VL_SHIFTR_WWI(256,256,8, __Vtemp_h5336a7a2__0, vlSelf->Tile__DOT__dcache__DOT__v, 
                  (0xffU & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                            >> 4U)));
    Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0 
        = (1U & __Vtemp_h5336a7a2__0[0U]);
    VL_SHIFTR_WWI(256,256,8, __Vtemp_hc380ab92__0, vlSelf->Tile__DOT__dcache__DOT__d, 
                  (0xffU & (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__dcache__DOT__is_dirty = ((IData)(Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0) 
                                                & __Vtemp_hc380ab92__0[0U]);
    vlSelf->Tile__DOT__dcache__DOT__hit = ((IData)(Tile__DOT__dcache__DOT____VdfgTmp_hf7d7bc41__0) 
                                           & (vlSelf->Tile__DOT__dcache__DOT___metaMem_tag_ext_RW0_rdata 
                                              == (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                  >> 0xcU)));
    vlSelf->Tile__DOT__dcache__DOT___T_34 = ((IData)(vlSelf->Tile__DOT__dcache__DOT__hit) 
                                             | (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg));
    vlSelf->Tile__DOT___dcache_io_cpu_resp_valid = 
        (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)))) 
               | (((1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__hit)) 
                  | ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__cpu_mask)))) 
                     & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc_reg)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__112(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__112\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEcall 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           & (IData)((0U == (0x10100000U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__113(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__113\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__isEbreak 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           & (IData)((0x100000U == (0x10100000U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__114(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__114\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__wen 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd)) 
           | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr_cmd) 
               >> 1U) & (0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                                         >> 0xfU)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__115(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__115\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__iaddrInvalid 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc_check) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
              >> 1U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__116(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__116\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__laddrInvalid 
        = (1U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)) 
                  | (2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)))
                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu
                  : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ld_type)) 
                     & (0U != (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__117(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__117\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__saddrInvalid 
        = (1U & ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type))
                  ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu
                  : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type)) 
                     & (0U != (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__118(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__118\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT___io_expt_T_12 
        = ((3U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                  >> 0x1cU)) > (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__PRV));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__119(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__119\n"); );
    // Body
    if (vlSelf->reset) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_pc = 0U;
    } else if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_pc 
            = (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc);
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__120(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__120\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out 
        = ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_mul
            : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))
                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_pe
                : vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__121(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__121\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__bypass__DOT____VdfgTmp_h8e44abc1__0 
        = ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)) 
           | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel)) 
              | (5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_sel))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__122(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__122\n"); );
    // Init
    VlWide<4>/*127:0*/ __Vtemp_h7f6b82f2__0;
    // Body
    if (vlSelf->reset) {
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst = 0x13U;
    } else if ((1U & (~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)))) {
        if (vlSelf->Tile__DOT__icache__DOT__is_alloc_reg) {
            __Vtemp_h7f6b82f2__0[0U] = (IData)(vlSelf->Tile__DOT__icache__DOT__refill_buf_0);
            __Vtemp_h7f6b82f2__0[1U] = (IData)((vlSelf->Tile__DOT__icache__DOT__refill_buf_0 
                                                >> 0x20U));
            __Vtemp_h7f6b82f2__0[2U] = (IData)(vlSelf->Tile__DOT__icache__DOT__refill_buf_1);
            __Vtemp_h7f6b82f2__0[3U] = (IData)((vlSelf->Tile__DOT__icache__DOT__refill_buf_1 
                                                >> 0x20U));
        } else if (vlSelf->Tile__DOT__icache__DOT__ren_reg) {
            __Vtemp_h7f6b82f2__0[0U] = vlSelf->Tile__DOT__icache__DOT__rdata[0U];
            __Vtemp_h7f6b82f2__0[1U] = vlSelf->Tile__DOT__icache__DOT__rdata[1U];
            __Vtemp_h7f6b82f2__0[2U] = vlSelf->Tile__DOT__icache__DOT__rdata[2U];
            __Vtemp_h7f6b82f2__0[3U] = vlSelf->Tile__DOT__icache__DOT__rdata[3U];
        } else {
            __Vtemp_h7f6b82f2__0[0U] = vlSelf->Tile__DOT__icache__DOT__rdata_buf[0U];
            __Vtemp_h7f6b82f2__0[1U] = vlSelf->Tile__DOT__icache__DOT__rdata_buf[1U];
            __Vtemp_h7f6b82f2__0[2U] = vlSelf->Tile__DOT__icache__DOT__rdata_buf[2U];
            __Vtemp_h7f6b82f2__0[3U] = vlSelf->Tile__DOT__icache__DOT__rdata_buf[3U];
        }
        vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
            = (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__started) 
                  | (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_inst_kill)) 
                 | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___brCond_io_taken)) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt))
                ? 0x13U : __Vtemp_h7f6b82f2__0[(3U 
                                                & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                   >> 2U))]);
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__123(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__123\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__lshift 
        = (vlSelf->Tile__DOT___dcache_io_cpu_resp_bits_data 
           >> (0x18U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
                        << 3U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__124(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__124\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc = vlSelf->__Vdly__Tile__DOT__core__DOT__dpath__DOT__pc;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__127(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__127\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__134(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__134\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__135(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__135\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_1 
        = (0x1ffffffffULL & (4ULL + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__136(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__136\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__started 
        = vlSelf->reset;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__137(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__137\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
            & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)) 
               & (IData)(vlSelf->Tile__DOT__dcache__DOT___T_34))) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__138(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__138\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT___T_35 = ((IData)(vlSelf->Tile__DOT__dcache__DOT___T_34) 
                                             | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__139(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__139\n"); );
    // Body
    if (vlSelf->Tile__DOT__icache__DOT__ren_reg) {
        vlSelf->Tile__DOT__icache__DOT__rdata_buf[0U] 
            = vlSelf->Tile__DOT__icache__DOT__rdata[0U];
        vlSelf->Tile__DOT__icache__DOT__rdata_buf[1U] 
            = vlSelf->Tile__DOT__icache__DOT__rdata[1U];
        vlSelf->Tile__DOT__icache__DOT__rdata_buf[2U] 
            = vlSelf->Tile__DOT__icache__DOT__rdata[2U];
        vlSelf->Tile__DOT__icache__DOT__rdata_buf[3U] 
            = vlSelf->Tile__DOT__icache__DOT__rdata[3U];
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__140(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__140\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__is_alloc_reg = vlSelf->Tile__DOT__icache__DOT__is_alloc;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__141(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__141\n"); );
    // Body
    if (vlSelf->reset) {
        vlSelf->__Vdly__Tile__DOT__icache__DOT__read_count = 0U;
    } else if (vlSelf->Tile__DOT__icache__DOT___T_22) {
        vlSelf->__Vdly__Tile__DOT__icache__DOT__read_count 
            = (1U & ((IData)(vlSelf->Tile__DOT__icache__DOT__read_count) 
                     - (IData)(1U)));
    }
    if (((IData)(vlSelf->Tile__DOT__icache__DOT___T_22) 
         & (~ (IData)(vlSelf->Tile__DOT__icache__DOT__read_count)))) {
        vlSelf->Tile__DOT__icache__DOT__refill_buf_0 
            = vlSelf->io_nasti_r_bits_data;
    }
    if (((IData)(vlSelf->Tile__DOT__icache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__read_count))) {
        vlSelf->Tile__DOT__icache__DOT__refill_buf_1 
            = vlSelf->io_nasti_r_bits_data;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__142(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__142\n"); );
    // Body
    if (vlSelf->Tile__DOT___icache_io_cpu_resp_valid) {
        vlSelf->Tile__DOT__icache__DOT__addr_reg = vlSelf->Tile__DOT___core_io_icache_req_bits_addr;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__143(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__143\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__144(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__144\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1 
        = ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                            >> 0xfU))) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__regs_ext__DOT__Memory
           [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                      >> 0xfU))] : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__145(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__145\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__146(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__146\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2 
        = ((0x73U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
           | (0x100073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__147(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__147\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10 
        = ((0x37U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
           | (0x17U == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__148(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__148\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75 
        = (0xfU == ((0xf00000U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                  >> 8U)) | (0xfffffU 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__149(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__149\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN = 
        ((0x380U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                    >> 5U)) | (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__153(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__153\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__155(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__155\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_alloc));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__156(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__156\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__157(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__157\n"); );
    // Body
    vlSelf->Tile__DOT___dcache_io_nasti_aw_valid = 
        ((0U != (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
         & ((1U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state))
             ? ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__hit)) 
                & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty))
             : ((2U == (IData)(vlSelf->Tile__DOT__dcache__DOT__state)) 
                & ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT___T_35)) 
                   & (IData)(vlSelf->Tile__DOT__dcache__DOT__is_dirty)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__159(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__159\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__read_count = vlSelf->__Vdly__Tile__DOT__icache__DOT__read_count;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__160(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__160\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT___T_22 = ((6U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                             & ((IData)(vlSelf->io_nasti_r_valid) 
                                                & (1U 
                                                   == (IData)(vlSelf->Tile__DOT__arb__DOT__state))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__161(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__161\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9 
        = ((0x6fU == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)) 
           | (0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__163(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__163\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__165(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__165\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild)
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_qvmask)
            : ((0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                                >> 0xfU))) ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory
               [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                          >> 0xfU))] : vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT__qvm_ext__DOT__Memory
               [(0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                          >> 0x14U))]));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__166(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__166\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1vaild)
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___bypass_io_rs1_out
            : vlSelf->Tile__DOT__core__DOT__dpath__DOT___regFile_io_rdata1);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__167(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__167\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2) 
           | (0x10000073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__168(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__168\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75) 
           | (0x100fU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__169(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__169\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14 
        = ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__170(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__170\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_1 
        = ((0x7c00U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                       >> 0x11U)) | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__171(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__171\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10) 
           | (0x6fU == (0x7fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__172(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__172\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3 
        = ((0x2f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x373U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x3f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__173(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__173\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20 
        = ((0x23U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0xa3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (0x123U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__174(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__174\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0x83U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x103U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x203U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | (0x283U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__175(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__175\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8 
        = ((0x63U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
           | ((0xe3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x263U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x2e3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | ((0x363U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                       | (0x3e3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__176(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__176\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0 
        = ((0x1fc00U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                        >> 0xfU)) | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__177(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__177\n"); );
    // Body
    vlSelf->io_nasti_ar_bits_addr = ((IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid)
                                      ? (0xfffffff0U 
                                         & vlSelf->Tile__DOT__dcache__DOT__addr_reg)
                                      : (0xfffffff0U 
                                         & vlSelf->Tile__DOT__icache__DOT__addr_reg));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__178(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__178\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT___wmask_T_2 = (0x7ffffU 
                                                  & ((IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask) 
                                                     << 
                                                     (0xcU 
                                                      & vlSelf->Tile__DOT__icache__DOT__addr_reg)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__179(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__179\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__ren_reg = vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en;
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__180(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__180\n"); );
    // Body
    vlSelf->io_nasti_aw_valid = ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__arb__DOT__state)))) 
                                 & (IData)(vlSelf->Tile__DOT___dcache_io_nasti_aw_valid));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__181(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__181\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_21 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__182(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__182\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__read_wrap_out = 
        ((IData)(vlSelf->Tile__DOT__icache__DOT___T_22) 
         & (IData)(vlSelf->Tile__DOT__icache__DOT__read_count));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__183(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__183\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign 
        = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
            >> 0x1fU) == (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
                          >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__184(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__184\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__185(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__185\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
           - vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__186(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__186\n"); );
    // Body
    vlSelf->__VdfgTmp_h8aef8152__0 = (1U & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_14)) 
                                            | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__187(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__187\n"); );
    // Init
    VlWide<8>/*255:0*/ __Vtemp_hd0742a16__0;
    // Body
    VL_SHIFTR_WWI(256,256,8, __Vtemp_hd0742a16__0, vlSelf->Tile__DOT__icache__DOT__v, 
                  (0xffU & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0 
        = (1U & __Vtemp_hd0742a16__0[0U]);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__188(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__188\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__189(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__189\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_23) 
           | (0x10200073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__190(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__190\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h67a27549__0) 
           | ((0x67U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__191(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__191\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22 
        = ((0x433U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
           | ((0x4b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
              | ((0x533U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                 | ((0x5b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                    | ((0x633U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                       | ((0x6b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                          | ((0x733U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                             | (0x7b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0))))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__192(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__192\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__193(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__193\n"); );
    // Body
    vlSelf->__VdfgTmp_hfd3c45f9__0 = ((0x333U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0) 
                                      | (0x3b3U == vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__195(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__195\n"); );
    // Init
    VlWide<8>/*255:0*/ __Vtemp_h0dd6d4d8__0;
    // Body
    VL_SHIFTR_WWI(256,256,8, __Vtemp_h0dd6d4d8__0, vlSelf->Tile__DOT__icache__DOT__d, 
                  (0xffU & (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                            >> 4U)));
    vlSelf->Tile__DOT__icache__DOT__is_dirty = ((IData)(vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0) 
                                                & __Vtemp_h0dd6d4d8__0[0U]);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__196(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__196\n"); );
    // Body
    vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0 
        = (1U & ((~ (IData)(vlSelf->io_nasti_aw_valid)) 
                 & (~ (IData)((0U != (IData)(vlSelf->Tile__DOT__arb__DOT__state))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__197(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__197\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__198(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__198\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__199(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__199\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__lt 
        = (1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign)
                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
                     >> 0x1fU) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_rs1_in0 
                                  >> 0x1fU)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__200(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__200\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__ltu 
        = (1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__isSameSign)
                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__brCond__DOT__diff 
                     >> 0x1fU) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2 
                                  >> 0x1fU)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__201(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__201\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__is_alloc = ((6U 
                                                 == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                                                & (IData)(vlSelf->Tile__DOT__icache__DOT__read_wrap_out));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__202(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__202\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5 
        = ((0x100fU == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__203(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__203\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__204(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__204\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__hit = ((IData)(vlSelf->Tile__DOT__icache__DOT____VdfgTmp_hf7d7bc41__0) 
                                           & ((((~ (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_1)) 
                                                & (IData)(vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN_0))
                                                ? vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT__Memory
                                               [vlSelf->Tile__DOT__icache__DOT__metaMem_tag_ext__DOT___GEN]
                                                : 0U) 
                                              == (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                  >> 0xcU)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__205(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__205\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__206(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__206\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_3) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_12));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__207(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__207\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__208(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__208\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__209(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__209\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_1_ext__RW0_wmask 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc)
            ? 0xfU : (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                              >> 4U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__210(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__210\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_2_ext__RW0_wmask 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc)
            ? 0xfU : (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                              >> 8U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__211(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__211\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_3_ext__RW0_wmask 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc)
            ? 0xfU : (0xfU & (vlSelf->Tile__DOT__icache__DOT___wmask_T_2 
                              >> 0xcU)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__212(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__212\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__dataMem_0_ext__RW0_wmask 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc)
            ? 0xfU : (0xfU & ((IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask) 
                              << (0xcU & vlSelf->Tile__DOT__icache__DOT__addr_reg))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__213(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__213\n"); );
    // Body
    if (vlSelf->Tile__DOT__icache__DOT__is_alloc) {
        vlSelf->Tile__DOT__icache__DOT__wdata[0U] = (IData)(vlSelf->Tile__DOT__icache__DOT__refill_buf_0);
        vlSelf->Tile__DOT__icache__DOT__wdata[1U] = (IData)(
                                                            (vlSelf->Tile__DOT__icache__DOT__refill_buf_0 
                                                             >> 0x20U));
        vlSelf->Tile__DOT__icache__DOT__wdata[2U] = (IData)(vlSelf->io_nasti_r_bits_data);
        vlSelf->Tile__DOT__icache__DOT__wdata[3U] = (IData)(
                                                            (vlSelf->io_nasti_r_bits_data 
                                                             >> 0x20U));
    } else {
        vlSelf->Tile__DOT__icache__DOT__wdata[0U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[1U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[2U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
        vlSelf->Tile__DOT__icache__DOT__wdata[3U] = vlSelf->Tile__DOT__icache__DOT__cpu_data;
    }
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__214(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__214\n"); );
    // Body
    vlSelf->Tile__DOT___arb_io_dcache_ar_ready = ((IData)(vlSelf->io_nasti_ar_ready) 
                                                  & (IData)(vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__215(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__215\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0)
            ? 0U : ((0x23U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                     ? 3U : ((0xa3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN))
                              ? 2U : (0x123U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__216(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__216\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__217(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__217\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT___T_35 = ((IData)(vlSelf->Tile__DOT__icache__DOT__hit) 
                                             | (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc_reg));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__218(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__218\n"); );
    // Body
    vlSelf->Tile__DOT___icache_io_cpu_resp_valid = 
        (1U & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
               | (((1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
                   & (IData)(vlSelf->Tile__DOT__icache__DOT__hit)) 
                  | ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__cpu_mask)))) 
                     & (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc_reg)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__219(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__219\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15) 
           | ((0xf3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
              | ((0x173U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                 | ((0x1f3U == (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN)) 
                    | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_13)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__220(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__220\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_17 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__221(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__221\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__222(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__222\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__223(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__223\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__224(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__224\n"); );
    // Body
    vlSelf->Tile__DOT___arb_io_icache_ar_ready = ((~ (IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid)) 
                                                  & (IData)(vlSelf->Tile__DOT___arb_io_dcache_ar_ready));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__225(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__225\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__226(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__226\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en 
        = (((2U == (IData)(vlSelf->Tile__DOT__icache__DOT__state)) 
            & (IData)(vlSelf->Tile__DOT__icache__DOT___T_35)) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__227(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__227\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT___icache_io_cpu_resp_valid)) 
                 | (~ (IData)(vlSelf->Tile__DOT___dcache_io_cpu_resp_valid))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__228(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__228\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__229(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__229\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__230(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__230\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__231(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__231\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hee3191ef__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__232(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__232\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_24 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_15));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__233(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__233\n"); );
    // Body
    vlSelf->io_nasti_ar_valid = (((IData)(vlSelf->Tile__DOT___dcache_io_nasti_ar_valid) 
                                  | (IData)(vlSelf->Tile__DOT___icache_io_nasti_ar_valid)) 
                                 & (IData)(vlSelf->Tile__DOT__arb__DOT____VdfgTmp_h4accae57__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__234(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__234\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__regFile__DOT___T_1 
        = (((~ ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall))) 
            & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__wb_en)) 
           & (0U != (0x1fU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_inst 
                              >> 7U))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__235(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__235\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__st_type)
            : (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_st_type));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__236(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__236\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en 
        = (1U & ((~ (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)) 
                 & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
                    & ((~ (IData)((0U != (IData)(vlSelf->Tile__DOT__icache__DOT__state)))) 
                       | (1U == (IData)(vlSelf->Tile__DOT__icache__DOT__state))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__237(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__237\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT__writeEnable = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en) 
                                                   & (IData)(vlSelf->Tile__DOT__icache__DOT__is_alloc));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__238(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__238\n"); );
    // Body
    vlSelf->Tile__DOT___core_io_dcache_req_valid = 
        ((~ (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)) 
         & (((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h34518d2f__0)) 
             & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_20)) 
            | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h17a479ec__0)) 
               & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__239(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__239\n"); );
    // Body
    vlSelf->__VdfgTmp_hbd242f9a__0 = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h5535d1e5__0) 
                                      | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__240(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__240\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hc5fed368__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___ctrlSignals_T_75));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__241(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__241\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___GEN 
        = ((0xccU & ((0xf0U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                               >> 4U)) | (0xcU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                                  >> 0xcU)))) 
           | (0x33U & ((0xf0U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                 >> 8U)) | (0xfU & 
                                            (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                             >> 0x10U)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__242(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__242\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__243(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__243\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__244(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__244\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__245(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__245\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__246(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__246\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT___ctrl_io_inst_kill 
        = ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_10)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_9) 
              | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_8)) 
                 & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_7) 
                    | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_6)) 
                       & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_5) 
                          | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_2)) 
                             & (0x10000073U == vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst))))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__247(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__247\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__248(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__248\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__249(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__249\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en 
        = ((~ (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)) 
           & ((IData)(vlSelf->Tile__DOT__dcache__DOT____VdfgTmp_hc92af663__0) 
              & (IData)(vlSelf->Tile__DOT___core_io_dcache_req_valid)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__250(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__250\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__251(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__251\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__252(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__252\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__253(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__253\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_hdfe1f980__0) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_22));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__254(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__254\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT___GEN_1 = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
                                              | (IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__255(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__255\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_en 
        = ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_rdata_MPORT_3_en) 
           | (IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__256(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__256\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0 
        = ((~ ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h5a0d4b50__0) 
               | ((7U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                  | ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                     | ((5U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                        | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)) 
                           | (3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)))))))) 
           & (2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_mul_op_out)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__257(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__257\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___next_pc_T_4 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT___ctrl_io_pc_sel)) 
           | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___brCond_io_taken));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__258(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__258\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__259(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__259\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en 
        = (1U & (~ ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__fe_reg_inst 
                     >> 0x19U) & ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h0a8baeef__0)) 
                                  & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h990c992e__0)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__260(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__260\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
        = (((~ ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h8bec0e33__0) 
                | (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_19))) 
            & ((IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_16) 
               | ((~ (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT___GEN_18)) 
                  & (IData)(vlSelf->Tile__DOT__core__DOT__ctrl__DOT____VdfgTmp_h28509d5e__0))))
            ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_br_rs2
            : vlSelf->Tile__DOT__core__DOT__dpath__DOT___immGen_io_out);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__261(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__261\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__262(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__262\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT___GEN_0 
        = ((((QData)((IData)((- (IData)(((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0) 
                                         & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                            >> 0x1fU)))))) 
             << 0x20U) | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1))) 
           * (((QData)((IData)((- (IData)(((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__mul__DOT____VdfgTmp_h2c378339__0) 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                              >> 0x1fU)))))) 
               << 0x20U) | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__263(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__263\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
        = (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
           + ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0)
               ? (- vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
               : vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__264(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__264\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT___shiftr_T_5 
        = (0x1ffffffffULL & VL_SHIFTRS_QQI(33,33,5, 
                                           (((QData)((IData)(
                                                             ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT____VdfgTmp_h2a774d50__0) 
                                                              & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin 
                                                                 >> 0x1fU)))) 
                                             << 0x20U) 
                                            | (QData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__alu__DOT__shin))), 
                                           (0x1fU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__265(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__265\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__266(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__266\n"); );
    // Body
    vlSelf->Tile__DOT___core_io_dcache_req_bits_mask 
        = (0xfU & ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T))
                    ? ((IData)(1U) << (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum))
                    : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T))
                        ? ((IData)(3U) << (3U & vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum))
                        : (- (IData)((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___io_dcache_req_bits_mask_T)))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__267(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__267\n"); );
    // Body
    vlSelf->__VdfgTmp_h92f16298__0 = (0x3fffffffU & 
                                      ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)
                                        ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__ew_reg_alu 
                                           >> 2U) : 
                                       (vlSelf->Tile__DOT__core__DOT__dpath__DOT___alu_io_sum 
                                        >> 2U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__268(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__268\n"); );
    // Body
    vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
        = ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__stall)
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc)
            : ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___csr_io_expt)
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
                                                    + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pc)))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__269(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__269\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__270(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__270\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__271(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__271\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__272(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__272\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in))
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__273(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__273\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__274(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__274\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xdU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__275(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__275\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_hbe180063__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 9U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__276(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__276\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha0dc5a79__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 5U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__277(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__277\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h6d2b06e7__0 
        = (1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 1U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__278(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__278\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8br8_pe_8br8_type 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? ((0xeU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                ? 5U : ((0xdU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                         ? 3U : ((0xcU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                  ? 1U : ((0xbU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                           ? 4U : (
                                                   (0xaU 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)) 
                                                   << 1U)))))
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__279(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__279\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
              >> 0xcU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__280(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__280\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h677344a9__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 0xdU))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__281(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__281\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7416c8a4__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
              >> 8U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__282(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__282\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h6743b8e4__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 9U))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__283(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__283\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h7407838c__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
              >> 4U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__284(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__284\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h675c1f6c__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 5U))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__285(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__285\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h747068eb__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__286(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__286\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h66af7931__0 
        = (1U & (~ ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                    & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                       >> 1U))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__287(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__287\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__288(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__288\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__289(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__289\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__290(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__290\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__291(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__291\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__292(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__292\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__293(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__293\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__294(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__294\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__295(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__295\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__296(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__296\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__297(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__297\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__298(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__298\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__299(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__299\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__300(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__300\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__301(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__301\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__302(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__302\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__303(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__303\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__304(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__304\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__305(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__305\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__306(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__306\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__307(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__307\n"); );
    // Body
    vlSelf->__VdfgTmp_h6297395b__0 = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                       ? ((9U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                           ? 4U : (
                                                   (8U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                                                    ? 2U
                                                    : 
                                                   (7U 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))))
                                       : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__308(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__308\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xfU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__309(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__309\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xdU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__310(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__310\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 0xbU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__311(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__311\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 9U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__312(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__312\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 7U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__313(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__313\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 5U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__314(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__314\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 3U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__315(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__315\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37 
        = (1U & ((~ ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                        >> 1U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_qnorm_Qvm_en)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__316(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__316\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__317(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__317\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__318(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__318\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__319(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__319\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__320(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__320\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__321(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__321\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__322(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__322\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__323(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__323\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__324(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__324\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? ((0x10U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
                ? 2U : (0xfU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out)))
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__325(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__325\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__326(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__326\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__327(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__327\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__328(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__328\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__329(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__329\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__330(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__330\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__331(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__331\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__332(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__332\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__333(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__333\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__334(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__334\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__335(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__335\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__336(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__336\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__337(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__337\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__338(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__338\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__339(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__339\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN 
        = ((vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
            >> 7U) & ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                      & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                         >> 0x1fU)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__340(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__340\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__341(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__341\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__342(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__342\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__343(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__343\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__344(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__344\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__345(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__345\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__346(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__346\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__347(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__347\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__348(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__348\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__349(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__349\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x1cU) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__350(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__350\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x18U) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__351(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__351\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__352(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__352\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x1cU) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__353(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__353\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__354(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__354\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__355(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__355\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__356(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__356\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__357(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__357\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__358(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__358\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__359(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__359\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__360(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__360\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__361(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__361\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__362(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__362\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__363(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__363\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__364(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__364\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__365(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__365\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__366(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__366\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__367(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__367\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x18U) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__368(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__368\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0 
        = ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__369(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__369\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__370(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__370\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__371(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__371\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__372(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__372\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0xfU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__373(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__373\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__374(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__374\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x17U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__375(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__375\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__376(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__376\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__377(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__377\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__378(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__378\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__379(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__379\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__380(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__380\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
               >> 0x1cU) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__381(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__381\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__382(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__382\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__383(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__383\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__384(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__384\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__385(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__385\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 0x1fU));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__386(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__386\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__387(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__387\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x18U) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__388(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__388\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__389(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__389\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                        >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__390(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__390\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__391(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__391\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
               >> 0x1cU) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__392(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__392\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x14U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__393(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__393\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0xcU)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__394(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__394\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 4U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__395(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__395\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x18U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__396(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__396\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 0x10U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__397(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__397\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                       >> 8U)) : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__398(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__398\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0 
        = ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__399(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__399\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xffU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__400(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__400\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
              >> 7U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__401(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__401\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2)
            : 0U);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__402(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__402\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op 
        = ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
            ? ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_pe_op_out))
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
                                                     : 0xaU))))))
            : 0xaU);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__403(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__403\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__icache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                                  >> 4U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__404(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__404\n"); );
    // Body
    vlSelf->Tile__DOT__icache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__icache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__icache__DOT__addr_reg 
                        >> 4U) : (vlSelf->Tile__DOT___core_io_icache_req_bits_addr 
                                  >> 4U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__405(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__405\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT___GEN_0 = (0xffU 
                                              & ((IData)(vlSelf->Tile__DOT__dcache__DOT__dataMem_3_MPORT_4_en)
                                                  ? 
                                                 (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                                                  >> 4U)
                                                  : 
                                                 (vlSelf->__VdfgTmp_h92f16298__0 
                                                  >> 2U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__406(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__406\n"); );
    // Body
    vlSelf->Tile__DOT__dcache__DOT____Vcellinp__metaMem_tag_ext__RW0_addr 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__dcache__DOT__writeEnable)
                     ? (vlSelf->Tile__DOT__dcache__DOT__addr_reg 
                        >> 4U) : (vlSelf->__VdfgTmp_h92f16298__0 
                                  >> 2U)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__407(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__407\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__408(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__408\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b14015__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__409(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__409\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645afe27__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__410(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__410\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_3 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3510e805__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__411(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__411\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_3 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67bb0233__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__412(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__412\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35f007dc__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__413(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__413\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h679bbde2__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__414(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__414\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15333__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__415(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__415\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h647964bc__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bcdc1__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__416(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__416\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77d60__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__417(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__417\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h64691316__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__418(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__418\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ec00__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__419(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__419\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8a6e__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__420(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__420\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_7 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 4U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                  >> 0x1cU) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__421(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__421\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_7 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 4U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x1cU) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__422(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__422\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_6 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35d82d3f__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__423(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__423\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_6 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646287c1__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__424(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__424\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35e15236__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__425(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__425\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h678bccc4__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__426(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__426\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35c77273__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__427(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__427\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h646ae801__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__428(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__428\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h3527ed03__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__429(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__429\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h67cd8b71__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__430(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__430\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs1B8_3 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 8U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                  >> 0x18U) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__431(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__431\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT__rs2B8_3 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 8U) | ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x18U) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__432(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__432\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__433(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__433\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645ec445__0))))
              : 0U) << 8U) | ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                  >> 0x18U) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__434(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__434\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN) 
            << 4U) | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                       ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                          >> 0x1cU) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__435(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__435\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___GEN) 
            << 8U) | ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                       ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                          >> 0x18U) : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__436(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__436\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h679bbde2__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__437(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__437\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67bb0233__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__438(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__438\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h645afe27__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__439(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__439\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64691316__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__440(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__440\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h647964bc__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bcdc1__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__441(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__441\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ae801__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__442(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__442\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h678bccc4__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__443(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__443\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__444(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__444\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0 
        = (0x7fffU & (- (IData)(((vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                  >> 7U) & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__445(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__445\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35f007dc__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__446(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__446\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3510e805__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__447(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__447\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b14015__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__448(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__448\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77d60__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__449(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__449\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15333__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__450(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__450\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0xbU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35c77273__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__451(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__451\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x13U)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35e15236__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__452(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__452\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 0x1bU)))) 
                         << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82d3f__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__453(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__453\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xfU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3588fd5f__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__454(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__454\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14 
        = ((((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
              ? (0xffU & (- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35b56e53__0))))
              : 0U) << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35d82a30__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__455(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__455\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde2a1c__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__456(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__456\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0x1f0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                           << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                     >> 0x17U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb72212__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__457(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__457\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bb3c9c9__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h386c5b78__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__458(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__458\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc7645__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__459(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__459\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0) 
                << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae73b__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__460(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__460\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0x1fff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                             << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                       >> 0xfU))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bde291b__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__461(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__461\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__462(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__462\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__463(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__463\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__464(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__464\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3 
        = (0x3ffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0) 
                       * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__465(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__465\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8a6e__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__466(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__466\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64691316__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__467(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__467\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bcdc1__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__468(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__468\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3 
        = (0x3ffffU & (((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0))) 
                                     << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                       * ((0x3ff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0))) 
                                       << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646284c2__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__469(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__469\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bc1eb15__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bcde11e__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__470(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__470\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0x1fffff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                               << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                         >> 7U))) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bbc4b5c__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__471(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__471\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67cd8b71__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__472(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__472\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__473(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__473\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ae801__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__474(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__474\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__475(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__475\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678bccc4__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__476(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__476\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__477(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__477\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646287c1__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__478(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__478\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7 
        = (0x3ffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0) 
                     * (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__479(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__479\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3ba00b14__0) 
                << 4U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3bad1931__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__480(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__480\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h679bbde2__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__481(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__481\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e4f0f0__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h678f4ee2__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h67bb0233__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__482(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__482\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d4fed2__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h647964bc__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645afe27__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__483(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__483\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7 
        = (0x3ffU & (((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b56e53__0))) 
                                 << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                     * ((0x3f0U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h645ec445__0))) 
                                   << 4U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h64525761__0))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__484(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__484\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xfffffff0U & (((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                << 4U) & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                          << 1U))) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h3b1ae426__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__485(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__485\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__486(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__486\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__487(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__487\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__488(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__488\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__489(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__489\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h35cb10ff__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__490(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__490\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__491(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__491\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xff00U & ((- (IData)((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646ef6e9__0))) 
                           << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8a6e__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__492(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__492\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0 
        = ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
            ? ((0xf0U & ((- (IData)(((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                     & (vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs2 
                                        >> 3U)))) << 4U)) 
               | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h67cd8b71__0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__493(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__493\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__494(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__494\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___br8x8_io_qvm_out 
        = ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
            ? ((0xc000U & ((((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
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
                                                              << 1U))))))))))
            : ((0x7000U & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__br8x8__DOT____VdfgTmp_h74277fc3__0) 
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
                                                         << 1U)))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__495(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__495\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1 
        = (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__496(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__496\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1 
        = (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__497(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__497\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__498(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__498\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__499(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__499\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__500(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__500\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__501(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__501\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__502(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__502\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__503(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__503\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7 
        = ((0x7fff00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                                           >> 7U)))) 
                         << 8U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__504(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__504\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__505(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__505\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__506(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__506\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13 
        = ((0x7fff0000U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                                             >> 0xfU)))) 
                           << 0x10U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__507(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__507\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0) 
            << 8U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__508(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__508\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16 
        = (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h4d0697d8__0) 
            << 0x10U) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__509(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__509\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_1) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__510(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__510\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_2) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__511(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__511\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_3) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__512(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__512\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_4) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__513(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__513\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_5) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__514(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__514\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_6) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__515(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__515\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55 
        = (0xffffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                       ? VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1)))
                       : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                              - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                           : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                                  + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                               : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                   ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                                      - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1))
                                   : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                       ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_1) 
                                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1))
                                       : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__516(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__516\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85 
        = (0xffffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                       ? VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2)))
                       : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                              - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                           : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                                  + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                               : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                   ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                                      - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2))
                                   : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                       ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_2) 
                                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2))
                                       : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__517(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__517\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                            - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                                + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0))
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9))
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_8) 
                                        + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_9))
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__518(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__518\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115 
        = (0xffffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                       ? VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0))
                       : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                              - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                           : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                                  + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0))
                               : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                   ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                                      - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15))
                                   : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                       ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_14) 
                                          + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_15))
                                       : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__519(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__519\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                    + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__520(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__520\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4 
        = (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0) 
                    - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__521(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__521\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1 
        = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                      + (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__522(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__522\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4 
        = (0xffffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0) 
                      - (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__523(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__523\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__524(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__524\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__525(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__525\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_0);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__526(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__526\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_1);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__527(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__527\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_2);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__528(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__528\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS8out_WIRE_3);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__529(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__529\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_0);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__530(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__530\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_1);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__531(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__531\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_2);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__532(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__532\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28 
        = ((0x3fc0000U & ((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3 
                                            >> 0x11U)))) 
                          << 0x12U)) | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU8out_WIRE_3);
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__533(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__533\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__534(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__534\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__535(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__535\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__536(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__536\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__537(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__537\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__538(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__538\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_5));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__539(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__539\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_6));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__540(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__540\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulS4out_WIRE_7));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__541(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__541\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_0));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__542(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__542\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_1));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__543(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__543\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__544(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__544\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__545(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__545\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__546(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__546\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_5));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__547(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__547\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_6));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__548(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__548\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17 
        = ((0x3fc00U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7) 
                                          >> 9U)))) 
                        << 0xaU)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulU4out_WIRE_7));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__549(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__549\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3 
        = (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__550(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__550\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3 
        = (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__551(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__551\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_7 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_10 
                                << (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h64525761__0))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_235)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__552(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__552\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_6 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_7 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_6)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_205)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__553(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__553\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_5 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_6 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_5)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_175)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__554(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__554\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_4 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_5 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_4)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_145)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__555(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__555\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_3 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_4 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_3)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_115)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__556(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__556\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_2 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_3 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_2)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_85)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__557(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__557\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_1 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_2 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_1)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_55)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__558(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__558\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_3 
        = (0x7fffffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16
                           : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_16 
                                  << (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h646287c1__0))
                               : ((0x7fff0000U & ((- (IData)(
                                                             (1U 
                                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115) 
                                                                 >> 0xfU)))) 
                                                  << 0x10U)) 
                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_115)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__559(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__559\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_2 
        = (0x7fffffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13
                           : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_13 
                                  << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_2)))
                               : ((0x7fff0000U & ((- (IData)(
                                                             (1U 
                                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85) 
                                                                 >> 0xfU)))) 
                                                  << 0x10U)) 
                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_85)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__560(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__560\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_1 
        = (0x7fffffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12
                           : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_12 
                                  << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_1)))
                               : ((0x7fff0000U & ((- (IData)(
                                                             (1U 
                                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55) 
                                                                 >> 0xfU)))) 
                                                  << 0x10U)) 
                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_55)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__561(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__561\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25 
        = (0xffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                     ? VL_SHIFTRS_III(8,8,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B4_0), 
                                      (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)))
                     : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4)
                         : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1)
                             : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                 ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_4)
                                 : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                     ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_1)
                                     : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__562(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__562\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25 
        = (0xffffU & ((8U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                       ? VL_SHIFTRS_III(16,16,4, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs1B8_0), 
                                        (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)))
                       : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4)
                           : ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1)
                               : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                   ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_4)
                                   : ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                                       ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_1)
                                       : 0U))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__563(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__563\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__564(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__564\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_1)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__565(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__565\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_24 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__566(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__566\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_28 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__567(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__567\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_23 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__568(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__568\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_27 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__569(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__569\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_22 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__570(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__570\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_26 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__571(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__571\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_18 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__572(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__572\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
        = (0x3ffffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25
                          : (VL_SHIFTRS_III(26,26,5, 
                                            (0x3ffffffU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_25 
                                                * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_19)), 
                                            (0x1fU 
                                             & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                                >> 0x11U))) 
                             + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_21)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__573(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__573\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_9 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__574(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__574\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_17 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__575(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__575\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_8 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__576(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__576\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_16 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__577(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__577\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_7 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__578(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__578\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_15 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__579(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__579\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_6 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__580(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__580\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_14 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__581(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__581\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_5 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__582(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__582\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_13 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__583(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__583\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_4 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__584(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__584\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_12 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__585(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__585\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_3 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__586(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__586\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_11 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__587(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__587\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__588(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__588\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
        = (0x3ffffU & ((0x100U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10
                        : (VL_SHIFTRS_III(18,18,5, 
                                          (0x3ffffU 
                                           & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_10 
                                              * vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_0)), 
                                          (0x1fU & 
                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm 
                                            >> 0x11U))) 
                           + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_2)));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__589(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__589\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
            ? (VL_LTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min8_T_3)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
            : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
                ? (VL_GTS_III(9, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                    ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max8_T_3)
                    : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__590(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__590\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out4b_Unorm_0 
        = (0x7fffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1
                         : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                             ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_1 
                                << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B4_0)))
                             : ((0x7fff00U & ((- (IData)(
                                                         (1U 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25) 
                                                             >> 7U)))) 
                                              << 8U)) 
                                | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out4b_Unorm_T_25)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__591(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__591\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__out8b_Unorm_0 
        = (0x7fffffffU & ((0xaU == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                           ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11
                           : ((6U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8alu8_pe_8alu8_op))
                               ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___GEN_11 
                                  << (0xfU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT__rs2B8_0)))
                               : ((0x7fff0000U & ((- (IData)(
                                                             (1U 
                                                              & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25) 
                                                                 >> 0xfU)))) 
                                                  << 0x10U)) 
                                  | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT___out8b_Unorm_T_25)))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__592(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__592\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_3)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__593(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__593\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_3)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__594(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__594\n"); );
    // Body
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
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__595(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__595\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___alu8x8_io_out 
        = ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
            ? ((((1U & ((~ ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
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
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ed03__0))))))))))
            : ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_ha22f1e8d__0)
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
                                             : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__alu8x8__DOT____VdfgTmp_h3527ec00__0)))))));
}
