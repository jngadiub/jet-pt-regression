import os, sys, time, math
from optparse import OptionParser
from array import array

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.colors as colors

from keras.models import load_model

from train_MLP_nomu import parse_config, get_features

import ROOT as rt
rt.gROOT.SetBatch(True)
import CMS_lumi, tdrstyle
tdrstyle.setTDRStyle()

CMS_lumi.writeExtraText = 1
CMS_lumi.extraText = "Run 3 Simulation"
CMS_lumi.lumi_sqrtS = "13 TeV" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)

iPeriod = 4

def getCanvas(name, lumi, iPos):

 CMS_lumi.lumi_13TeV = lumi

 if( iPos==0 ): CMS_lumi.relPosX = 0.12
 else: CMS_lumi.relPosX = 0.045
 
 H_ref = 600; 
 W_ref = 800; 
 W = W_ref
 H  = H_ref

 # references for T, B, L, R
 T = 0.08*H_ref
 B = 0.12*H_ref 
 L = 0.12*W_ref
 R = 0.04*W_ref

 canvas = rt.TCanvas(name,name,50,50,W,H)
 canvas.SetFillColor(0)
 canvas.SetBorderMode(0)
 canvas.SetFrameFillStyle(0)
 canvas.SetFrameBorderMode(0)
 canvas.SetLeftMargin( L/W )
 canvas.SetRightMargin( R/W )
 canvas.SetTopMargin( T/H )
 canvas.SetBottomMargin( B/H )
 canvas.SetTickx(0)
 canvas.SetTicky(0)
 
 return canvas

def get_pave_text(x1,y1,x2,y2):
 
    pt = rt.TPaveText(x1,y1,x2,y2,"NDC")
    pt.SetTextFont(42)
    pt.SetTextSize(0.04)
    pt.SetTextAlign(12)
    pt.SetFillColor(0)
    pt.SetBorderSize(0)
    pt.SetFillStyle(0)
    
    return pt
    
def get_legend(objs,labels,type='LP',pos='top right'):

 l = 0.
 if pos=='top right':
  l = rt.TLegend(0.62,0.75,0.80,0.90)
 elif pos=='top left':
  l = rt.TLegend(0.18,0.8,0.42,0.92) 
 
 l.SetTextSize(0.04)
 l.SetBorderSize(0)
 l.SetLineColor(1)
 l.SetLineStyle(1)
 l.SetLineWidth(1)
 l.SetFillColor(0)
 l.SetFillStyle(0)
 l.SetTextFont(42)
 for i,o in enumerate(objs):
  l.AddEntry(o,labels[i],type)
 
 return l

def draw(canv,legend,objs,draw_opts,label,xtitle="",ytitle="",xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=0):

    canv.cd()
    
    objs[0].Draw(draw_opts)
    if ymin != 0 or ymax != 0: objs[0].GetYaxis().SetRangeUser(ymin,ymax)
    if xmin != 0 or xmax != 0: objs[0].GetXaxis().SetRangeUser(xmin,xmax)
    objs[0].GetYaxis().SetTitle(ytitle)
    objs[0].GetXaxis().SetTitle(xtitle)
    objs[0].GetYaxis().SetTitleOffset(1.1)
    for o in range(1,len(objs)): objs[o].Draw(draw_opts+"same")
    if legend: legend.Draw()
    if pt: pt.Draw()
    
    CMS_lumi.CMS_lumi(canv, iPeriod, 0)
    canv.cd()
    canv.Update()
    canv.RedrawAxis()
    frame = canv.GetFrame()
    frame.Draw()
    formats = ['pdf','png','C','root']
    for f in formats: canv.SaveAs(options.outputDir+'/%s.%s'%(label,f))
    canv.Delete()
    del canv

def get_multi_graphs(graphs,gnames,colors,mstyles):

  mg = rt.TMultiGraph("mg","mg")

  for i,g in enumerate(graphs):
    g.SetName(gnames[i])
    g.SetMarkerColor(colors[i])
    g.SetLineColor(colors[i])
    g.SetMarkerStyle(mstyles[i])
    mg.Add(g)
  
  return mg
       
def fitResolutionWithDCB(h2dRes,options,which):

 rt.gSystem.Load("libHiggsAnalysisCombinedLimit")
 
 mean = array('d',[])
 meanErr = array('d',[])
 width = array('d',[])
 widthErr = array('d',[])
 x = array('d',[])
 xErr = array('d',[])
    
 for bin in range(1,h2dRes.GetNbinsX()+1):
    
    tmp=h2dRes.ProjectionY("q",bin,bin)
    tmpX=h2dRes.ProjectionX("qX",bin,bin)
    startbin   = 0.
    maxbin = 0.
    maxcontent = 0.
    x.append(tmpX.GetBinCenter(bin))
    xErr.append(0.)
    binningx = []
    for b in range(tmp.GetXaxis().GetNbins()):
      binningx.append(tmp.GetXaxis().GetBinLowEdge(b))
      if tmp.GetXaxis().GetBinCenter(b+1) > startbin and tmp.GetBinContent(b+1)>maxcontent:
        maxbin = b
        maxcontent = tmp.GetBinContent(b+1)
    tmpmean = tmp.GetXaxis().GetBinCenter(maxbin)
    tmpwidth = 0.5

    var = rt.RooRealVar("pt","pt",200,tmp.GetXaxis().GetXmin(),tmp.GetXaxis().GetXmax())
    var.setBinning(rt.RooBinning(len(binningx)-1,array("d",binningx)))
    
    MEAN	= rt.RooRealVar("mean_bin_%i"%(bin)  ,"mean_bin_%i"%(bin)  ,tmpmean , tmpmean-tmpwidth,tmpmean+tmpwidth)
    SIGMA	= rt.RooRealVar("sigma_bin_%i"%(bin) ,"sigma_bin_%i"%(bin) ,tmpwidth, tmpwidth*0.02, tmpwidth*0.10)	     
    ALPHA1	= rt.RooRealVar("alpha1_bin_%i"%(bin) ,"alpha1_bin_%i"%(bin),1.2,0.0,18)
    ALPHA2	= rt.RooRealVar("alpha2_bin_%i"%(bin),"alpha2_bin_%i"%(bin),1.2,0.0,10)
    N1  	= rt.RooRealVar("n1_bin_%i"%(bin)    ,"n1_bin_%i"%(bin)    ,5,0,600)
    N2  	= rt.RooRealVar("n2_bin_%i"%(bin)    ,"n2_bin_%i"%(bin)    ,5,0,50)

    function = rt.RooDoubleCB("model_bin_%i"%bin, "model_bin_%i"%bin, var, MEAN,SIGMA,ALPHA1,N1,ALPHA2,N2)
    
    cList = rt.RooArgList()
    cList.add(var)
    dataHist=rt.RooDataHist("data_bin_%i"%bin,"data_bin_%i"%bin,cList,tmp)
    
    fitresult = function.fitTo(dataHist,rt.RooFit.Save(1))
    fitresult.Print()
    
    plt = var.frame()
    dataHist.plotOn(plt)
    function.plotOn(plt)
    
    c1 =rt.TCanvas("c","",800,800)
    plt.Draw()
    c1.SaveAs(options.outputDir+"/debug_fitres_%s_bin%i.png"%(which,bin))
    
    tmpmean = MEAN.getVal()
    tmpmeanErr = MEAN.getError()
    tmpwidth = SIGMA.getVal()
    tmpwidthErr = SIGMA.getError()
    
    mean.append(tmpmean)
    meanErr.append(tmpmeanErr)
    width.append(tmpwidth)
    widthErr.append(tmpwidthErr) 
    
 return x,xErr,mean,meanErr,width,widthErr    


def fitResolution(h2dRes,options,which):

 mean = array('d',[])
 meanErr = array('d',[])
 width = array('d',[])
 widthErr = array('d',[])
 x = array('d',[])
 xErr = array('d',[])
    
 for bin in range(1,h2dRes.GetNbinsX()+1):
    
    tmp=h2dRes.ProjectionY("q",bin,bin)
    tmpX=h2dRes.ProjectionX("qX",bin,bin)
    startbin   = 0.
    maxbin = 0.
    maxcontent = 0.
    x.append(tmpX.GetBinCenter(bin))
    xErr.append(0.)
    for b in range(tmp.GetXaxis().GetNbins()):
      if tmp.GetXaxis().GetBinCenter(b+1) > startbin and tmp.GetBinContent(b+1)>maxcontent:
        maxbin = b
        maxcontent = tmp.GetBinContent(b+1)
    tmpmean = tmp.GetXaxis().GetBinCenter(maxbin)
    tmpwidth = 0.5
    g1 = rt.TF1("g1","gaus", tmpmean-tmpwidth,tmpmean+tmpwidth)
    tmp.Fit(g1, "SR")
    #c1 =rt.TCanvas("c","",800,800)
    #tmp.Draw()
    #c1.SaveAs(options.outputDir+"/debug_fitres_%s_bin%i.png"%(which,bin))
    tmpmean = g1.GetParameter(1)
    tmpwidth = g1.GetParameter(2)
    g1 = rt.TF1("g1","gaus", tmpmean-(tmpwidth*2),tmpmean+(tmpwidth*2))
    tmp.Fit(g1, "SR")
    c1 =rt.TCanvas("c","",800,800)
    tmp.Draw()
    c1.SaveAs(options.outputDir+"/debug_fitres_%s_bin%i.png"%(which,bin))
    tmpmean = g1.GetParameter(1)
    tmpmeanErr = g1.GetParError(1)
    tmpwidth = g1.GetParameter(2)
    tmpwidthErr = g1.GetParError(2)
    mean.append(tmpmean)
    meanErr.append(tmpmeanErr)
    width.append(tmpwidth)
    widthErr.append(tmpwidthErr) 
    
 return x,xErr,mean,meanErr,width,widthErr    

def getPredictions(X_test,yamlConfig,model):
    
    y_predict = model.predict(X_test)[:,0] #y = L1/reco (ratio) or reco/L1 (inv_ratio) or reco_pt (pT)
    print("Predict values shape:",y_predict.shape)
    
    l1_pt_test = X_test[:,0]
    if (yamlConfig['RegressTo'] == 'ratio'):
     y_predict[y_predict==0] = 1
     l1_pt_predicted = l1_pt_test/y_predict
    elif (yamlConfig['RegressTo'] == 'inv_ratio'): l1_pt_predicted = y_predict*l1_pt_test
    elif (yamlConfig['RegressTo'] == 'diff'): l1_pt_predicted = y_predict+l1_pt_test
    elif (yamlConfig['RegressTo'] == 'inv_diff'): l1_pt_predicted = l1_pt_test-y_predict
    else: l1_pt_predicted = y_predict
    
    return l1_pt_predicted
     
def makeTriggerTurnOn(l1_pt_test,l1_pt_predicted,reco_pt_test,thresholds,options):

    hreco_den = rt.TH1F("hreco_den","hreco_den",20,0,1000)
    hrecoTrue_num = rt.TH1F("hrecoTrue_num","hrecoTrue_num",20,0,1000)
    hrecoPred_num = rt.TH1F("hrecoPred_num","hrecoPred_num",20,0,1000)
    hreco_den_zoom = rt.TH1F("hreco_den_zoom","hreco_den_zoom",150,0,300)
    hrecoTrue_num_zoom = rt.TH1F("hrecoTrue_num_zoom","hrecoTrue_num_zoom",150,0,300)
    hrecoPred_num_zoom = rt.TH1F("hrecoPred_num_zoom","hrecoPred_num_zoom",150,0,300)

    for i,v in enumerate(reco_pt_test):
     hreco_den.Fill(v)
     hreco_den_zoom.Fill(v)
     if l1_pt_test[i] > thresholds[0]:
      hrecoTrue_num.Fill(v)
      hrecoTrue_num_zoom.Fill(v)
     if l1_pt_predicted[i] > thresholds[1]:
      hrecoPred_num.Fill(v)
      hrecoPred_num_zoom.Fill(v)
              
    effTrue = rt.TEfficiency(hrecoTrue_num,hreco_den) 
    effTrue.SetName("effTrue")
    effTrue.SetLineColor(rt.kBlack)
    effTrue.SetMarkerColor(rt.kBlack)
    effTrue.SetMarkerStyle(20)
    effPred = rt.TEfficiency(hrecoPred_num,hreco_den) 
    effPred.SetName("effPred")
    effPred.SetLineColor(rt.kBlue)
    effPred.SetMarkerColor(rt.kBlue)
    effPred.SetMarkerStyle(24)

    effTrue_zoom = rt.TEfficiency(hrecoTrue_num_zoom,hreco_den_zoom) 
    effTrue_zoom.SetName("effTrue_zoom")
    effTrue_zoom.SetLineColor(rt.kBlack)
    effTrue_zoom.SetMarkerColor(rt.kBlack)
    effTrue_zoom.SetMarkerStyle(20)
    effPred_zoom = rt.TEfficiency(hrecoPred_num_zoom,hreco_den_zoom) 
    effPred_zoom.SetName("effPred_zoom")
    effPred_zoom.SetLineColor(rt.kBlue)
    effPred_zoom.SetMarkerColor(rt.kBlue)
    effPred_zoom.SetMarkerStyle(24)
    
    l = rt.TLegend(0.6203008,0.7491289,0.8007519,0.8972125)
    l.SetTextSize(0.04) #0.05
    l.SetBorderSize(0)
    l.SetLineColor(1)
    l.SetLineStyle(1)
    l.SetLineWidth(1)
    l.SetFillColor(0)
    l.SetFillStyle(0)
    l.SetTextFont(42)
    l.AddEntry(effTrue_zoom,'Standard corrections','LP')
    l.AddEntry(effPred_zoom,'DNN corrections','LP')

    c = getCanvas('c', '', 0)
    c.cd()
    frame = c.DrawFrame(0,0,1000,1.4)
    frame.GetYaxis().SetTitleOffset(0.9)
    frame.GetYaxis().SetTitle('Efficiency')
    frame.GetXaxis().SetTitle('Offline Jet p_{T} [GeV]')
    effTrue.Draw('same')
    effPred.Draw('same')
    l.Draw()
    
    CMS_lumi.CMS_lumi(c, iPeriod, 0)
    c.cd()
    c.Update()
    c.RedrawAxis()
    frame = c.GetFrame()
    frame.Draw()
       
    c.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff.pdf'%(options.label))
    c.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff.png'%(options.label))
    c.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff.C'%(options.label))
    c.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff.root'%(options.label))
    effTrue.SaveAs(options.outputDir+"/%s_singlejet_"%options.label+effTrue.GetName()+".root")
    effPred.SaveAs(options.outputDir+"/%s_singlejet_"%options.label+effPred.GetName()+".root")
        
    c_zoom = getCanvas('c_zoom', '', 0)
    c_zoom.cd()
    frame_zoom = c_zoom.DrawFrame(0,0,300,1.4)
    frame_zoom.GetYaxis().SetTitleOffset(0.9)
    frame_zoom.GetYaxis().SetTitle('Efficiency')
    frame_zoom.GetXaxis().SetTitle('Offline Jet p_{T} [GeV]')
    effTrue_zoom.Draw('same')
    effPred_zoom.Draw('same')
    l.Draw()
    
    CMS_lumi.CMS_lumi(c_zoom, iPeriod, 0)
    c_zoom.cd()
    c_zoom.Update()
    c_zoom.RedrawAxis()
    frame = c_zoom.GetFrame()
    frame.Draw()
       
    c_zoom.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff_ptzoom.pdf'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff_ptzoom.png'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff_ptzoom.C'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_singlejet_trigger_eff_ptzoom.root'%(options.label))
    effTrue_zoom.SaveAs(options.outputDir+"/%s_singlejet_"%options.label+effTrue_zoom.GetName()+".root")
    effPred_zoom.SaveAs(options.outputDir+"/%s_singlejet_"%options.label+effPred_zoom.GetName()+".root")

def makeRates(options,yamlConfig,model):

    if options.rateFile == '': return 0,0

    print(":::MakeRates::::")

    h5File = h5py.File(options.rateFile,'r')

    eventTree = h5File['eventInfo'][()]
    eventNames = ["event", "nPV", "isoMu", "nRecoJets","nL1Jets"]
    event_df = pd.DataFrame(eventTree,columns=eventNames)
    events_val = event_df.values
        
    l1jetTree = h5File['l1Jet'][()]
    print("L1 jet tree shape:",l1jetTree.shape)    
    all_features = h5File['l1JetNames'][()]
    all_features = [f.decode('utf-8') for f in all_features]
    print("All inputs:")
    print(all_features)      

    # List of features to use
    features = yamlConfig['Inputs']
    print("Inputs to use:")
    print(features)

    features_df = pd.DataFrame(l1jetTree,columns=all_features)
    l1_pt = features_df[["pT"]].values   
    l1_pt = l1_pt[:,0]
        
    if features[0] == 'RawEt': features_df[["RawEt"]] = features_df[["RawEt"]].values/2. - features_df[["PUEt"]].values/2.
    features_df = features_df[features]
    features_val = features_df.values

    features_val = features_val[:1000000,:]
    l1_pt = l1_pt[:1000000]
    
    features_val = features_val[np.where(l1_pt < 1023),:]
    print(features_val.shape)
    features_val = np.reshape(features_val,(features_val.shape[1],features_val.shape[2]))
    print(features_val.shape)
    #features_val = features_val[0]
    l1_pt = l1_pt[np.where(l1_pt < 1023)]
    print(l1_pt.shape)
    
    plt.figure()
    plt.semilogy()
    plt.hist(l1_pt,200,facecolor='green',alpha=0.75)   
    plt.xlabel('Standard L1 jet pT ')
    plt.savefig(options.outputDir+'/l1jetpt_standard_corrs_makerates.png')
        
    print("predict...")	
    l1_pt_predict = getPredictions(features_val,yamlConfig,model)

    nJetBins = 500
    jetLo = 0.
    jetHi = 500.
    jetBinWidth = (jetHi-jetLo)/nJetBins  
  
    singleJetRates_emu = rt.TH1F("singleJetRates_emu", "singleJetRates_emu", nJetBins, jetLo, jetHi)
    singleJetRates_regression = rt.TH1F("singleJetRates_regression", "singleJetRates_regression", nJetBins, jetLo, jetHi)
   
    nEvents = l1_pt.shape[0]#1000000#l1jetTree.shape[0]
    for e in range(nEvents):
   
        if e%100000 == 0: print("Event",e,"/",nEvents)
	
        jetEt_1 = l1_pt[e]
        new_jetEt_1 = l1_pt_predict[e]
    	  
        for bin in range(nJetBins):       

           if jetEt_1 >= jetLo + (bin*jetBinWidth):
              singleJetRates_emu.Fill(jetLo+(bin*jetBinWidth))
           if new_jetEt_1 >= jetLo + (bin*jetBinWidth):
              singleJetRates_regression.Fill(jetLo+(bin*jetBinWidth))     

    instLumi = 2e34 
    mbXSec   = 6.92e-26
    norm = instLumi * mbXSec / nEvents #Hz/cm^2 * cm^2
   
    if not options.is_data:
     singleJetRates_emu.Scale(norm)
     singleJetRates_regression.Scale(norm)
    singleJetRates_emu.SetLineColor(rt.kBlack)
    singleJetRates_regression.SetLineColor(rt.kBlue)
    singleJetRates_emu.SetLineWidth(2)
    singleJetRates_regression.SetLineWidth(2)

    l = rt.TLegend(0.6203008,0.7491289,0.8007519,0.8972125)
    l.SetTextSize(0.04) #0.05
    l.SetBorderSize(0)
    l.SetLineColor(1)
    l.SetLineStyle(1)
    l.SetLineWidth(1)
    l.SetFillColor(0)
    l.SetFillStyle(0)
    l.SetTextFont(42)
    l.AddEntry(singleJetRates_emu,'Standard correction','L')
    l.AddEntry(singleJetRates_regression,'DNN correction','L')
   
    c = getCanvas('c', '', 0)
    c.cd()
    c.SetLogy()

    singleJetRates_emu.GetXaxis().SetTitle('Jet E_{T} threshold [GeV]')
    singleJetRates_emu.GetYaxis().SetTitle('Trigger rate [Hz]')
    singleJetRates_emu.GetYaxis().SetTitleOffset(1.0)
    singleJetRates_emu.Draw("HIST")
    singleJetRates_regression.Draw("HISTsame")
    l.Draw()
   
    CMS_lumi.CMS_lumi(c, iPeriod, 0)
    c.cd()
    c.Update()
    c.RedrawAxis()
    frame = c.GetFrame()
    frame.Draw()
    c.SaveAs(options.outputDir+'/rates.pdf')
    c.SaveAs(options.outputDir+'/rates.png')
    c.SaveAs(options.outputDir+'/rates.C')
    c.SaveAs(options.outputDir+'/rates.root')
    singleJetRates_emu.SaveAs(options.outputDir+"/"+singleJetRates_emu.GetName()+".root")
    singleJetRates_regression.SaveAs(options.outputDir+"/"+singleJetRates_regression.GetName()+".root")
   
    cut = 9999
    for ix in range(1,singleJetRates_emu.GetNbinsX()+1):
   	   if singleJetRates_emu.GetBinContent(ix) <= options.rateValue:
   		   cut = singleJetRates_emu.GetXaxis().GetBinLowEdge(ix)
   		   break
    cut_regression = 9999
    for ix in range(1,singleJetRates_regression.GetNbinsX()+1):
   	   if singleJetRates_regression.GetBinContent(ix) <= options.rateValue:
   		    cut_regression = singleJetRates_regression.GetXaxis().GetBinLowEdge(ix)
   		    break
   
    print("Cut at fixed single jet rate",options.rateValue,"before regression",cut)
    print("Cut at fixed single jet rate",options.rateValue,"after regression",cut_regression)
  
    return cut, cut_regression

def makeResolution(reco_pt_test,reco_eta_test,l1_ptcorr,l1_pt_predict,options,which="pt"):

    ptbins = [20,25,30,35,40,45,50,55,60,70,80,90,100,120,140,160,200,250,300]
    binsy = []
    minx=-2.
    maxx=1.
    for i in range(0,100): binsy.append(minx+i*(maxx-minx)/100.)
    binsy_resp = []
    minx=0
    maxx=3.
    for i in range(0,100): binsy_resp.append(minx+i*(maxx-minx)/100.)
        
    if which == 'eta':
     hres = rt.TH2F("hres","hres",25,-5.0,5.0,100,-2,1)
     hres_regression = rt.TH2F("hres_regression","hres_regression",25,-5.0,5.0,100,-2,1)
     hresp = rt.TH2F("hresp","hresp",25,-5.0,5.0,100,0,3)
     hresp_regression = rt.TH2F("hresp_regression","hresp_regression",25,-5.0,5.0,100,0,3)
    else:
     hres = rt.TH2F("hres","hres",len(ptbins)-1,array('f',ptbins),len(binsy)-1,array('f',binsy))
     hres_regression = rt.TH2F("hres_regression","hres_regression",len(ptbins)-1,array('f',ptbins),len(binsy)-1,array('f',binsy))
     hresp = rt.TH2F("hresp","hresp",len(ptbins)-1,array('f',ptbins),len(binsy_resp)-1,array('f',binsy_resp))
     hresp_regression = rt.TH2F("hresp_regression","hresp_regression",len(ptbins)-1,array('f',ptbins),len(binsy_resp)-1,array('f',binsy_resp))
     
    hres_1D = rt.TH1F("hres_1D","hres_1D",len(binsy)-1,array('f',binsy))
    hres_1D_regression = rt.TH1F("hres_1D_regression","hres_1D_regression",len(binsy)-1,array('f',binsy))
    hresp_1D = rt.TH1F("hresp_1D","hresp_1D",len(binsy_resp)-1,array('f',binsy_resp))
    hresp_1D_regression = rt.TH1F("hresp_1D_regression","hres_1D_regression",len(binsy_resp)-1,array('f',binsy_resp))
             
    resolution_predicted = (reco_pt_test[:]-l1_pt_predict[:])/reco_pt_test[:]
    resolution_corr = (reco_pt_test[:]-l1_ptcorr[:])/reco_pt_test[:] #this uses JECs corrected L1 pT   
    resp_predicted = l1_pt_predict[:]/reco_pt_test[:]
    resp_corr = l1_ptcorr[:]/reco_pt_test[:]
            
    if which=='eta':	    
     for i,v in enumerate(reco_eta_test):
      hres_regression.Fill(v,resolution_predicted[i])
      hres.Fill(v,resolution_corr[i])
      hresp_regression.Fill(v,resp_predicted[i])
      hresp.Fill(v,resp_corr[i])
    else:
     for i,v in enumerate(reco_pt_test):
      hres_regression.Fill(v,resolution_predicted[i])
      hres.Fill(v,resolution_corr[i])
      hresp_regression.Fill(v,resp_predicted[i])
      hresp.Fill(v,resp_corr[i])
      
    for i,v in enumerate(resolution_predicted):
     #if reco_pt_test[i] > 50. and math.fabs(reco_eta_test[i]) > 3.0:
     hres_1D_regression.Fill(resolution_predicted[i])
     hres_1D.Fill(resolution_corr[i])
     hresp_1D_regression.Fill(resp_predicted[i])
     hresp_1D.Fill(resp_corr[i])            
    	               
    c = getCanvas('c_2D', '', 0)
    c.cd()
    c.SetLeftMargin(0.15)
    if which == 'eta': draw(c,0,[hres],"COLZ",'resolution_vs_%s_2D'%which,xtitle='Offline Jet #eta',ytitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}')
    else: draw(c,0,[hres],"COLZ",'resolution_vs_%s_2D'%which,xtitle='Offline Jet p_{T}',ytitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}')
    
    c_regression = getCanvas('c_2D_regression', '', 0)
    c_regression.cd()
    c_regression.SetLeftMargin(0.15)
    if which == 'eta': draw(c_regression,0,[hres_regression],"COLZ",'resolution_regression_vs_%s_2D'%which,xtitle='Offline Jet #eta',ytitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}')
    else: draw(c_regression,0,[hres_regression],"COLZ",'resolution_regression_vs_%s_2D'%which,xtitle='Offline Jet p_{T}',ytitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}')

    c_resp = getCanvas('c_2D_resp', '', 0)
    c_resp.cd()
    c_resp.SetLeftMargin(0.15)
    if which == 'eta': draw(c_resp,0,[hresp],"COLZ",'response_vs_%s_2D'%which,xtitle='Offline Jet #eta',ytitle='p_{T}^{L1} / p_{T}^{reco}')
    else: draw(c_resp,0,[hresp],"COLZ",'response_vs_%s_2D'%which,xtitle='Offline Jet p_{T}',ytitle='p_{T}^{L1} / p_{T}^{reco}')

    c_resp_regression = getCanvas('c_2D_resp_regression', '', 0)
    c_resp_regression.cd()
    c_resp_regression.SetLeftMargin(0.15)
    if which == 'eta': draw(c_resp_regression,0,[hresp_regression],"COLZ",'response_regression_vs_%s_2D'%which,xtitle='Offline Jet #eta',ytitle='p_{T}^{L1} / p_{T}^{reco}')
    else: draw(c_resp_regression,0,[hresp_regression],"COLZ",'response_regression_vs_%s_2D'%which,xtitle='Offline Jet p_{T}',ytitle='p_{T}^{L1} / p_{T}^{reco}')
                
    #x, xErr, mean, meanErr, width, widthErr = fitResolution(hres,options,which)	
    #x, xErr, mean_regression, meanErr_regression, width_regression, widthErr_regression = fitResolution(hres_regression,options,which+"regression")
    #x, xErr, mean_regression, meanErr_regression, width_regression, widthErr_regression = fitResolutionWithDCB(hres_regression,options,which+"regression")
    #print(mean)
    #print(width)
    #print(mean_regression)
    #print(width_regression)
    
    x = array('d',[])
    xErr = array('d',[])
    mean = array('d',[])
    meanErr = array('d',[])
    width = array('d',[])
    widthErr = array('d',[])
    ratio = array('d',[])
    ratioErr = array('d',[])
    mean_regression = array('d',[])
    meanErr_regression = array('d',[])
    width_regression = array('d',[])
    widthErr_regression = array('d',[])
    ratio_regression = array('d',[])
    ratioErr_regression = array('d',[])
    respMean_regression = array('d',[])
    respMeanErr_regression = array('d',[])
    respWidth_regression = array('d',[])
    respWidthErr_regression = array('d',[])
    respMean = array('d',[])
    respMeanErr = array('d',[])
    respWidth = array('d',[])
    respWidthErr = array('d',[])
                
    for b in range(1,hres.GetNbinsX()+1):
    
     proj = hres.ProjectionY("p",b,b)
     proj_regression = hres_regression.ProjectionY("p_regression",b,b)
     proj_resp = hresp.ProjectionY("presp",b,b)
     proj_resp_regression = hresp_regression.ProjectionY("presp_regression",b,b)
          
     x.append(hres.GetXaxis().GetBinCenter(b))
     xErr.append(0.)

     width.append(proj.GetStdDev())
     widthErr.append(proj.GetStdDevError())
     mean.append(proj.GetMean())
     meanErr.append(proj.GetMeanError())
     width_regression.append(proj_regression.GetStdDev())
     widthErr_regression.append(proj_regression.GetStdDevError())
     mean_regression.append(proj_regression.GetMean())
     meanErr_regression.append(proj_regression.GetMeanError())

     respMean_regression.append(proj_resp_regression.GetMean())
     respMeanErr_regression.append(proj_resp_regression.GetMeanError())
     respWidth_regression.append(proj_resp_regression.GetStdDev())
     respWidthErr_regression.append(proj_resp_regression.GetStdDevError())
     respMean.append(proj_resp.GetMean())
     respMeanErr.append(proj_resp.GetMeanError())
     respWidth.append(proj_resp.GetStdDev())
     respWidthErr.append(proj_resp.GetStdDevError())
    
     m = 1.0 - proj.GetMean()
     s = proj.GetStdDev()
     r = 0.
     if m!=0: r = s/m
     sm = proj.GetMeanError()
     ss = proj.GetStdDevError()
     err = 0.
     if m!=0 and s!=0: err = r*math.sqrt(sm**2/m**2+ss**2/s**2)
     ratio.append(r)
     ratioErr.append(err)
     
     print("Before regression:",b,s,m,ss,sm,r)

     m = 1.0 - proj_regression.GetMean()
     s = proj_regression.GetStdDev()
     r = 0.
     if m!=0: r = s/m
     sm = proj_regression.GetMeanError()
     ss = proj_regression.GetStdDevError()
     err=0.
     if m!=0 and s!=0: err = r*math.sqrt(sm**2/m**2+ss**2/s**2) 
     ratio_regression.append(r)
     ratioErr_regression.append(err)   
     print("After regression:",b,s,m,ss,sm,r)
       
     #debugging plots
     c1 = getCanvas('c1', '', 0)
     c1.cd()
     c1.SetBottomMargin(0.15)
     proj.SetLineColor(rt.kBlack)
     proj.SetLineWidth(2)
     if proj.Integral()!=0: proj.Scale(1./proj.Integral())
     proj_regression.SetLineColor(rt.kBlue)
     proj_regression.SetLineWidth(2)
     if proj_regression.Integral()!=0: proj_regression.Scale(1./proj_regression.Integral())
     if proj.GetMaximum() < proj_regression.GetMaximum(): proj.SetMaximum(proj_regression.GetMaximum()*1.2)
     l = get_legend([proj,proj_regression],['Standard correction','DNN correction'],'L','top left') 
     pt = get_pave_text(0.18,0.67,0.42,0.8)
     text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(mean[-1],width[-1]))
     text.SetTextColor(rt.kBlack)
     text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(mean_regression[-1],width_regression[-1]))
     text.SetTextColor(rt.kBlue)
     draw(c1,l,[proj,proj_regression],"HIST","debug_fitres_%s_bin%i"%(which,b),xtitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}',ytitle='',xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=pt)

    #### FINAL PLOTS #####
    
    #plot mean resolution          
    g_mean = rt.TGraphErrors(len(x),x,mean,xErr,meanErr)    
    g_regression_mean = rt.TGraphErrors(len(x),x,mean_regression,xErr,meanErr_regression)  
    mg_mean = get_multi_graphs([g_mean,g_regression_mean],["g_mean","g_regression_mean"],[rt.kBlack,rt.kBlue],[20,20])

    l = get_legend([g_mean,g_regression_mean],['Standard correction','DNN correction'])            
    c_final_mean = getCanvas('c_final_mean', '', 0)
    c_final_mean.SetLeftMargin(0.15)
    if which == 'eta': draw(c_final_mean,l,[mg_mean],"AP",'mean_resolution_vs_%s_1D'%which,xtitle='Offline Jet #eta',ytitle='#mu [ ( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=-1.0,ymax=0.5)
    else: draw(c_final_mean,l,[mg_mean],"AP",'mean_resolution_vs_%s_1D'%which,xtitle='Offline Jet p_{T}',ytitle='#mu [ ( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=-0.5,ymax=0.5)
    
    #plot sigma resolution
    g_sigma = rt.TGraphErrors(len(x),x,width,xErr,widthErr)    
    g_regression_sigma = rt.TGraphErrors(len(x),x,width_regression,xErr,widthErr_regression)  
    mg_sigma = get_multi_graphs([g_sigma,g_regression_sigma],["g_sigma","g_regression_sigma"],[rt.kBlack,rt.kBlue],[20,20])

    l = get_legend([g_sigma,g_regression_sigma],['Standard correction','DNN correction'])            
    c_final_sigma = getCanvas('c_final_sigma', '', 0)
    c_final_sigma.SetLeftMargin(0.15)
    if which == 'eta': draw(c_final_sigma,l,[mg_sigma],"AP",'sigma_resolution_vs_%s_1D'%which,xtitle='Offline Jet #eta',ytitle='#sigma [ ( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=1.0)
    else: draw(c_final_sigma,l,[mg_sigma],"AP",'sigma_resolution_vs_%s_1D'%which,xtitle='Offline Jet p_{T}',ytitle='#sigma [ ( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=1.0)
      
    #plot sigma/mean resolution
    g_ratio = rt.TGraphErrors(len(x),x,ratio,xErr,ratioErr)    
    g_regression_ratio = rt.TGraphErrors(len(x),x,ratio_regression,xErr,ratioErr_regression)  
    mg_ratio = get_multi_graphs([g_ratio,g_regression_ratio],["g_raio","g_regression_ratio"],[rt.kBlack,rt.kBlue],[20,20])

    l = get_legend([g_ratio,g_regression_ratio],['Standard correction','DNN correction'])            
    c_final_ratio = getCanvas('c_final_ratio', '', 0)
    c_final_ratio.SetLeftMargin(0.15)
    if which == 'eta': draw(c_final_ratio,l,[mg_ratio],"AP",'ratio_resolution_vs_%s_1D'%which,xtitle='Offline Jet #eta',ytitle='#sigma / (1-#mu)',xmin=0.,xmax=0.,ymin=0.1,ymax=0.6)
    else: draw(c_final_ratio,l,[mg_ratio],"AP",'ratio_resolution_vs_%s_1D'%which,xtitle='Offline Jet p_{T}',ytitle='#sigma / (1-#mu)',xmin=0.,xmax=0.,ymin=0.1,ymax=0.6)

    #mean response
    g_resp_mean = rt.TGraphErrors(len(x),x,respMean,xErr,respMeanErr)    
    g_regression_resp_mean = rt.TGraphErrors(len(x),x,respMean_regression,xErr,respMeanErr_regression)          
    mg_resp_mean = get_multi_graphs([g_resp_mean,g_regression_resp_mean],["g_resp_mean","g_regression_resp_mean"],[rt.kBlack,rt.kBlue],[20,20])

    l = get_legend([g_resp_mean,g_regression_resp_mean],['Standard correction','DNN correction'])            
    c_final_resp_mean = getCanvas('c_final_resp_mean', '', 0)
    c_final_resp_mean.SetLeftMargin(0.15)
    if which == 'eta': draw(c_final_resp_mean,l,[mg_resp_mean],"AP",'mean_response_vs_%s_1D'%which,xtitle='Offline Jet #eta',ytitle='#mu [ p_{T}^{L1} / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=2.)
    else: draw(c_final_resp_mean,l,[mg_resp_mean],"AP",'mean_response_vs_%s_1D'%which,xtitle='Offline Jet p_{T}',ytitle='#mu [ p_{T}^{L1} / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=2.)

    #sigma response 
    g_resp_sigma = rt.TGraphErrors(len(x),x,respWidth,xErr,respWidthErr)    
    g_regression_resp_sigma = rt.TGraphErrors(len(x),x,respWidth_regression,xErr,respWidthErr_regression)  
    mg_resp_sigma = get_multi_graphs([g_resp_sigma,g_regression_resp_sigma],["g_resp_sigma","g_regression_resp_sigma"],[rt.kBlack,rt.kBlue],[20,20])

    l = get_legend([g_resp_sigma,g_regression_resp_sigma],['Standard correction','DNN correction'])            
    c_final_resp_sigma = getCanvas('c_final_resp_sigma', '', 0)
    c_final_resp_sigma.SetLeftMargin(0.15)
    if which == 'eta': draw(c_final_resp_sigma,l,[mg_resp_sigma],"AP",'sigma_response_vs_%s_1D'%which,xtitle='Offline Jet #eta',ytitle='#sigma [ p_{T}^{L1} / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=1.)
    else: draw(c_final_resp_sigma,l,[mg_resp_sigma],"AP",'sigma_response_vs_%s_1D'%which,xtitle='Offline Jet p_{T}',ytitle='#sigma [ p_{T}^{L1} / p_{T}^{reco} ]',xmin=0.,xmax=0.,ymin=0.,ymax=1.)
    
    #global 1D response
    c_1D_resp = getCanvas('c_1D_resp','',0)
    c_1D_resp.cd()
    c_1D_resp.SetBottomMargin(0.15)    
    hresp_1D.SetLineColor(rt.kBlack)
    hresp_1D.SetLineWidth(2)
    hresp_1D_regression.SetLineColor(rt.kBlue)
    hresp_1D_regression.SetLineWidth(2)
    pt = get_pave_text(0.62,0.6,0.75,0.72)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hresp_1D.GetMean(),hresp_1D.GetStdDev()))
    text.SetTextColor(rt.kBlack)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hresp_1D_regression.GetMean(),hresp_1D_regression.GetStdDev()))
    text.SetTextColor(rt.kBlue)
    l = get_legend([hresp_1D,hresp_1D_regression],['Standard correction','DNN correction'],'L') 
    draw(c_1D_resp,l,[hresp_1D_regression,hresp_1D],"HIST",'response',xtitle='p_{T}^{L1} / p_{T}^{reco}',ytitle='a.u.',xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=pt)

    #global 1D resolution
    c_1D_res = getCanvas('c_1D_res','',0)
    c_1D_res.cd()
    c_1D_res.SetBottomMargin(0.15)
    hres_1D.SetLineColor(rt.kBlack)
    hres_1D.SetLineWidth(2)
    hres_1D_regression.SetLineColor(rt.kBlue)
    hres_1D_regression.SetLineWidth(2)
    l = get_legend([hres_1D,hres_1D_regression],['Standard correction','DNN correction'],'L','top left') 
    pt = get_pave_text(0.18,0.67,0.42,0.8)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hres_1D.GetMean(),hres_1D.GetStdDev()))
    text.SetTextColor(rt.kBlack)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hres_1D_regression.GetMean(),hres_1D_regression.GetStdDev()))
    text.SetTextColor(rt.kBlue)
    draw(c_1D_res,l,[hres_1D_regression,hres_1D],"HIST",'resolution',xtitle='( p_{T}^{reco}-p_{T}^{L1} ) / p_{T}^{reco}',ytitle='a.u.',xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=pt)

def makeHTResolution(n_matched_jets,reco_pt_test,reco_eta_test,l1_ptcorr,l1_eta,l1_pt_predict,options):

    hres_1D = rt.TH1F("hres_1D","hres_1D",200,-2.,2.)
    hres_1D_regression = rt.TH1F("hres_1D_regression","hres_1D_regression",200,-2.,2.)
    hresp_1D = rt.TH1F("hresp_1D","hresp_1D",100,0,3)
    hresp_1D_regression = rt.TH1F("hresp_1D_regression","hres_1D_regression",100,0,3)
    
    i=0
    for e,n in enumerate(n_matched_jets):
   
     if e%100000 == 0: print("Event",e,"/",n_matched_jets.shape[0])
     #if e > 10: break
     
     #print(e,n)
     
     HTreco = 0.
     HTL1 = 0.
     HTL1predict = 0.
     
     #if n==0: print(e,i,n)
     
     for j in range(n):
     
       #print("-------->",i+j,j,i)
       if reco_pt_test[i+j] > 30. and math.fabs(reco_eta_test[i+j]) < 2.5: HTreco+=reco_pt_test[i+j]
       if l1_ptcorr[i+j] > 30. and math.fabs(l1_eta[i+j]) < 2.5: HTL1+=l1_ptcorr[i+j]
       if l1_pt_predict[i+j] > 30. and math.fabs(l1_eta[i+j]) < 2.5: HTL1predict+=l1_pt_predict[i+j]

       #print(e,i,j,i+j,n,reco_pt_test[i+j],reco_eta_test[i+j],HTreco)
          
     i+=n
       
     if HTreco > 30. and HTL1 > 30.:
      hres_1D.Fill((HTreco-HTL1)/HTreco)
      hresp_1D.Fill(HTL1/HTreco)
     if HTreco > 30. and HTL1predict > 30.:
      hres_1D_regression.Fill((HTreco-HTL1predict)/HTreco)
      hresp_1D_regression.Fill(HTL1predict/HTreco)

    #global 1D response
    c_1D_resp = getCanvas('c_1D_resp','',0)
    c_1D_resp.cd()
    c_1D_resp.SetBottomMargin(0.15)    
    hresp_1D.SetLineColor(rt.kBlack)
    hresp_1D.SetLineWidth(2)
    hresp_1D_regression.SetLineColor(rt.kBlue)
    hresp_1D_regression.SetLineWidth(2)
    pt = get_pave_text(0.62,0.6,0.75,0.72)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hresp_1D.GetMean(),hresp_1D.GetStdDev()))
    text.SetTextColor(rt.kBlack)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hresp_1D_regression.GetMean(),hresp_1D_regression.GetStdDev()))
    text.SetTextColor(rt.kBlue)
    l = get_legend([hresp_1D,hresp_1D_regression],['Standard correction','DNN correction'],'L') 
    draw(c_1D_resp,l,[hresp_1D,hresp_1D_regression],"HIST",'HTresponse',xtitle='H_{T}^{L1} / H_{T}^{reco}',ytitle='a.u.',xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=pt)

    #global 1D resolution
    c_1D_res = getCanvas('c_1D_res','',0)
    c_1D_res.cd()
    c_1D_res.SetBottomMargin(0.15)
    hres_1D.SetLineColor(rt.kBlack)
    hres_1D.SetLineWidth(2)
    hres_1D_regression.SetLineColor(rt.kBlue)
    hres_1D_regression.SetLineWidth(2)
    l = get_legend([hres_1D,hres_1D_regression],['Standard correction','DNN correction'],'L','top left') 
    pt = get_pave_text(0.18,0.67,0.42,0.8)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hres_1D.GetMean(),hres_1D.GetStdDev()))
    text.SetTextColor(rt.kBlack)
    text = pt.AddText("#mu = %.2f, #sigma = %.2f"%(hres_1D_regression.GetMean(),hres_1D_regression.GetStdDev()))
    text.SetTextColor(rt.kBlue)
    draw(c_1D_res,l,[hres_1D,hres_1D_regression],"HIST",'HTresolution',xtitle='( H_{T}^{reco}-H_{T}^{L1} ) / H_{T}^{reco}',ytitle='a.u.',xmin=0.,xmax=0.,ymin=0.,ymax=0.,pt=pt)

def makeHTRates(options,yamlConfig,model):

    print(":::MakeHTRates::::")

    h5File = h5py.File(options.rateHTFile,'r')

    eventTree = h5File['eventInfo'][()]
    eventNames = h5File['eventNames'][()]
    eventNames = [n.decode('utf-8') for n in eventNames]
    print(eventNames)
    event_df = pd.DataFrame(eventTree,columns=eventNames)
    nL1Jets = event_df['nL1Jets'].values
        
    l1jetTree = h5File['l1Jet'][()]
    print("L1 jet tree shape:",l1jetTree.shape)    
    all_features = h5File['l1JetNames'][()]
    all_features = [f.decode('utf-8') for f in all_features]
    print("All inputs:")
    print(all_features)      

    # List of features to use
    features = yamlConfig['Inputs']
    print("Inputs to use:")
    print(features)

    features_df = pd.DataFrame(l1jetTree,columns=all_features)
    l1_pt = features_df[["pT"]].values   
    l1_pt = l1_pt[:,0]
    l1_eta = features_df[["eta"]].values 
    l1_eta = l1_eta[:,0]
    if features[0] == 'RawEt': features_df[["RawEt"]] = features_df[["RawEt"]].values/2. - features_df[["PUEt"]].values/2.
    features_df = features_df[features]
    features_val = features_df.values

    l1_pt_predict = getPredictions(features_val,yamlConfig,model)

    nHTBins = 970
    HTLo = 30.
    HTHi = 1000.
    HTBinWidth = (HTHi-HTLo)/nHTBins  
  
    HTRates_emu = rt.TH1F("HTRates_emu", "HTRates_emu", nHTBins, HTLo, HTHi)
    HTRates_regression = rt.TH1F("HTRates_regression", "HTRates_regression", nHTBins, HTLo, HTHi)
   
    nEvents = l1jetTree.shape[0]
    print("DEBUG nJets",nEvents,nL1Jets.shape)
    HT = 0.
    HTpredict = 0.
    thisEvent = 0
    for e in range(nEvents):
   
        #if e > 29: sys.exit()
        #print("*********",thisEvent,e,nL1Jets[thisEvent],nL1Jets[e])	
        if e%100000 == 0: print("Event",e,"/",nEvents)
        if e < thisEvent+nL1Jets[thisEvent]:
	
         if l1_pt[e] > 30. and math.fabs(l1_eta[e]) < 2.5: HT+=l1_pt[e]
         if l1_pt_predict[e] > 30. and math.fabs(l1_eta[e]) < 2.5: HTpredict+=l1_pt_predict[e]
         #print(e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1_pt[e],l1_pt_predict[e],l1_eta[e],HT,HTpredict)
	 	 
        else:
	
         #print("FILL HISTOS AND RESET ---->",HT,HTpredict,e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent])
         for bin in range(nHTBins):       

           if HT >= HTLo + (bin*HTBinWidth):
              HTRates_emu.Fill(HTLo+(bin*HTBinWidth))
           if HTpredict >= HTLo + (bin*HTBinWidth):
              HTRates_regression.Fill(HTLo+(bin*HTBinWidth))  
	      
	 #fill histo and reset
         thisEvent = e
         HT=0. #reset 
         HTpredict=0.
         if l1_pt[e] > 30. and math.fabs(l1_eta[e]) < 2.5: HT=l1_pt[e]
         if l1_pt_predict[e] > 30. and math.fabs(l1_eta[e]) < 2.5: HTpredict=l1_pt_predict[e]
         #print("::::ELSE::::",e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1_pt[e],l1_pt_predict[e],l1_eta[e],HT,HTpredict)

    instLumi = 2e34 
    mbXSec   = 6.92e-26
    norm = instLumi * mbXSec / nEvents #Hz/cm^2 * cm^2
   
    if not options.is_data:
     HTRates_emu.Scale(norm)
     HTRates_regression.Scale(norm)
    HTRates_emu.SetLineColor(rt.kBlack)
    HTRates_regression.SetLineColor(rt.kBlue)
    HTRates_emu.SetLineWidth(2)
    HTRates_regression.SetLineWidth(2)

    l = rt.TLegend(0.6203008,0.7491289,0.8007519,0.8972125)
    l.SetTextSize(0.04) #0.05
    l.SetBorderSize(0)
    l.SetLineColor(1)
    l.SetLineStyle(1)
    l.SetLineWidth(1)
    l.SetFillColor(0)
    l.SetFillStyle(0)
    l.SetTextFont(42)
    l.AddEntry(HTRates_emu,'Standard correction','L')
    l.AddEntry(HTRates_regression,'DNN correction','L')
   
    c = getCanvas('c', '', 0)
    c.cd()
    c.SetLogy()

    HTRates_emu.GetXaxis().SetTitle('H_{T} threshold [GeV]')
    HTRates_emu.GetYaxis().SetTitle('Trigger rate [Hz]')
    HTRates_emu.GetYaxis().SetTitleOffset(1.0)
    HTRates_emu.Draw("HIST")
    HTRates_regression.Draw("HISTsame")
    l.Draw()
   
    CMS_lumi.CMS_lumi(c, iPeriod, 0)
    c.cd()
    c.Update()
    c.RedrawAxis()
    frame = c.GetFrame()
    frame.Draw()
    c.SaveAs(options.outputDir+'/HTrates.pdf')
    c.SaveAs(options.outputDir+'/HTrates.png')
    c.SaveAs(options.outputDir+'/HTrates.C')
    c.SaveAs(options.outputDir+'/HTrates.root')
    HTRates_emu.SaveAs(options.outputDir+"/"+HTRates_emu.GetName()+".root")
    HTRates_regression.SaveAs(options.outputDir+"/"+HTRates_regression.GetName()+".root")
   
    if options.rateHTValue!=0:
     cut = 9999
     for ix in range(1,HTRates_emu.GetNbinsX()+1):
   	   if HTRates_emu.GetBinContent(ix) <= options.rateHTValue:
   		   cut = HTRates_emu.GetXaxis().GetBinLowEdge(ix)
   		   break
     cut_regression = 9999
     for ix in range(1,HTRates_regression.GetNbinsX()+1):
   	   if HTRates_regression.GetBinContent(ix) <= options.rateHTValue:
   		    cut_regression = HTRates_regression.GetXaxis().GetBinLowEdge(ix)
   		    break
   
     print("Cut at fixed HT rate",options.rateHTValue,"before regression",cut)
     print("Cut at fixed HT rate",options.rateHTValue,"after regression",cut_regression)
    
    if options.HTthreshold != 0:
     bin = HTRates_emu.GetXaxis().FindBin(options.HTthreshold)
     fixed_rate = HTRates_emu.GetBinContent(bin)
     cut = options.HTthreshold
     cut_regression = 9999
     for ix in range(1,HTRates_regression.GetNbinsX()+1):
   	   if HTRates_regression.GetBinContent(ix) <= fixed_rate:
   		    cut_regression = HTRates_regression.GetXaxis().GetBinLowEdge(ix)
   		    break

     print("Cut at fixed HT rate",fixed_rate,"before regression",cut)
     print("Cut at fixed HT rate",fixed_rate,"after regression",cut_regression)
       
    return cut, cut_regression

def makeHTTriggerTurnOn(n_matched_jets,l1_pt_test,l1_pt_predicted,reco_pt_test,l1_eta,reco_eta_test,thresholds,options):

    hreco_den = rt.TH1F("hreco_den","hreco_den",50,0,600)
    hrecoTrue_num = rt.TH1F("hrecoTrue_num","hrecoTrue_num",50,0,600)
    hrecoPred_num = rt.TH1F("hrecoPred_num","hrecoPred_num",50,0,600)
    hreco_den_zoom = rt.TH1F("hreco_den_zoom","hreco_den_zoom",150,0,300)
    hrecoTrue_num_zoom = rt.TH1F("hrecoTrue_num_zoom","hrecoTrue_num_zoom",150,0,300)
    hrecoPred_num_zoom = rt.TH1F("hrecoPred_num_zoom","hrecoPred_num_zoom",150,0,300)

    i=0
    for e,n in enumerate(n_matched_jets):
   
     if e%100000 == 0: print("Event",e,"/",n_matched_jets.shape[0])
     #if e > 10: break
     
     #print(e,n)
     
     HTreco = 0.
     HTL1 = 0.
     HTL1predict = 0.
     
     #if n==0: print(e,i,n)
     
     for j in range(n):
     
       #print("-------->",i+j,j,i)
       if reco_pt_test[i+j] > 30. and math.fabs(reco_eta_test[i+j]) < 2.5: HTreco+=reco_pt_test[i+j]
       if l1_ptcorr[i+j] > 30. and math.fabs(l1_eta[i+j]) < 2.5: HTL1+=l1_ptcorr[i+j]
       if l1_pt_predict[i+j] > 30. and math.fabs(l1_eta[i+j]) < 2.5: HTL1predict+=l1_pt_predict[i+j]

       #print(e,i,j,i+j,n,reco_pt_test[i+j],reco_eta_test[i+j],HTreco)
          
     i+=n
       
     hreco_den.Fill(HTreco)
     hreco_den_zoom.Fill(HTreco)
     if HTL1 > thresholds[0]:
       hrecoTrue_num.Fill(HTreco)
       hrecoTrue_num_zoom.Fill(HTreco)
     if HTL1predict > thresholds[1]:  
       hrecoPred_num.Fill(HTreco)
       hrecoPred_num_zoom.Fill(HTreco)      

              
    effTrue = rt.TEfficiency(hrecoTrue_num,hreco_den) 
    effTrue.SetName("effTrue")
    effTrue.SetLineColor(rt.kBlack)
    effTrue.SetMarkerColor(rt.kBlack)
    effTrue.SetMarkerStyle(20)
    effPred = rt.TEfficiency(hrecoPred_num,hreco_den) 
    effPred.SetName("effPred")
    effPred.SetLineColor(rt.kBlue)
    effPred.SetMarkerColor(rt.kBlue)
    effPred.SetMarkerStyle(24)

    effTrue_zoom = rt.TEfficiency(hrecoTrue_num_zoom,hreco_den_zoom) 
    effTrue_zoom.SetName("effTrue_zoom")
    effTrue_zoom.SetLineColor(rt.kBlack)
    effTrue_zoom.SetMarkerColor(rt.kBlack)
    effTrue_zoom.SetMarkerStyle(20)
    effPred_zoom = rt.TEfficiency(hrecoPred_num_zoom,hreco_den_zoom) 
    effPred_zoom.SetName("effPred_zoom")
    effPred_zoom.SetLineColor(rt.kBlue)
    effPred_zoom.SetMarkerColor(rt.kBlue)
    effPred_zoom.SetMarkerStyle(24)
    
    l = rt.TLegend(0.6203008,0.7491289,0.8007519,0.8972125)
    l.SetTextSize(0.04) #0.05
    l.SetBorderSize(0)
    l.SetLineColor(1)
    l.SetLineStyle(1)
    l.SetLineWidth(1)
    l.SetFillColor(0)
    l.SetFillStyle(0)
    l.SetTextFont(42)
    l.AddEntry(effTrue_zoom,'Standard corrections','LP')
    l.AddEntry(effPred_zoom,'DNN corrections','LP')

    c = getCanvas('c', '', 0)
    c.cd()
    frame = c.DrawFrame(0,0,600,1.4)
    frame.GetYaxis().SetTitleOffset(0.9)
    frame.GetYaxis().SetTitle('Efficiency')
    frame.GetXaxis().SetTitle('Offline H_{T} [GeV]')
    effTrue.Draw('same')
    effPred.Draw('same')
    l.Draw()
    
    CMS_lumi.CMS_lumi(c, iPeriod, 0)
    c.cd()
    c.Update()
    c.RedrawAxis()
    frame = c.GetFrame()
    frame.Draw()
       
    c.SaveAs(options.outputDir+'/%s_ht_trigger_eff.pdf'%(options.label))
    c.SaveAs(options.outputDir+'/%s_ht_trigger_eff.png'%(options.label))
    c.SaveAs(options.outputDir+'/%s_ht_trigger_eff.C'%(options.label))
    c.SaveAs(options.outputDir+'/%s_ht_trigger_eff.root'%(options.label))
    effTrue.SaveAs(options.outputDir+"/%s_ht_"%options.label+effTrue.GetName()+".root")
    effPred.SaveAs(options.outputDir+"/%s_ht_"%options.label+effPred.GetName()+".root")
        
    c_zoom = getCanvas('c_zoom', '', 0)
    c_zoom.cd()
    frame_zoom = c_zoom.DrawFrame(0,0,300,1.4)
    frame_zoom.GetYaxis().SetTitleOffset(0.9)
    frame_zoom.GetYaxis().SetTitle('Efficiency')
    frame_zoom.GetXaxis().SetTitle('Offline H_{T} [GeV]')
    effTrue_zoom.Draw('same')
    effPred_zoom.Draw('same')
    l.Draw()
    
    CMS_lumi.CMS_lumi(c_zoom, iPeriod, 0)
    c_zoom.cd()
    c_zoom.Update()
    c_zoom.RedrawAxis()
    frame = c_zoom.GetFrame()
    frame.Draw()
       
    c_zoom.SaveAs(options.outputDir+'/%s_ht_trigger_eff_ptzoom.pdf'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_ht_trigger_eff_ptzoom.png'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_ht_trigger_eff_ptzoom.C'%(options.label))
    c_zoom.SaveAs(options.outputDir+'/%s_ht_trigger_eff_ptzoom.root'%(options.label))
    effTrue_zoom.SaveAs(options.outputDir+"/%s_ht_"%options.label+effTrue_zoom.GetName()+".root")
    effPred_zoom.SaveAs(options.outputDir+"/%s_ht_"%options.label+effPred_zoom.GetName()+".root")
        	 	                                                           
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('--rateFile','--rateFile',action='store',type='string',dest='rateFile',default='',help='input h5 file for single jet rates')
    parser.add_option('--rateHTFile','--rateHTFile',action='store',type='string',dest='rateHTFile',default='',help='input h5 file for HT rates')
    parser.add_option('--rateValue','--rateValue',action='store',type=float,dest='rateValue',default=0,help='Fixed single jet rate for turn ons')
    parser.add_option('--rateHTValue','--rateHTValue',action='store',type=float,dest='rateHTValue',default=0,help='Fixed HT rate for turn ons')
    parser.add_option('-t','--threshold',action='store',type=float,dest='threshold',default=0,help='Fixed single jet threshold for turn ons')
    parser.add_option('-T','--HTthreshold',action='store',type=float,dest='HTthreshold',default=0,help='Fixed HT threshold for turn ons')
    parser.add_option('-i','--input',action='store',type='string',dest='inputFile',default='L1Ntuple.h5',help='input h5 file')
    parser.add_option('-m','--model',action='store',type='string',dest='model',default='train_MLP/KERAS_check_best_model.h5',help='trained model')
    parser.add_option('-o','--output',action='store',type='string',dest='outputDir',default='train_MLP',help='output directory')
    parser.add_option('-c','--config',action='store',type='string',dest='config',default='train_MLP.yml',help='configuration file')
    parser.add_option('-l','--label',action='store',type='string',dest='label',default='',help='Label for plots')
    parser.add_option("-s",'--split',action="store_true",dest="split", help="split in test and train",default=False)
    parser.add_option("--is_data",'--is_data',action="store_true",dest="is_data", help="Run over real data",default=False)
    (options,args) = parser.parse_args()

    if options.is_data: CMS_lumi.extraText = "Run 2 Data"

    print("Input file:",options.inputFile)
    print("Input model:",options.model)
    print("Outdir:",options.outputDir)
    print("Single jet rate file:",options.rateFile)
    print("HT rate file:",options.rateHTFile)
    print("Single jet rate value:",options.rateValue)
    print("HT rate value:",options.rateHTValue)
    print("Single jet threshold:",options.threshold)
    print("HT threshold:",options.HTthreshold)

    if os.path.isdir(options.outputDir):
        input("Warning: output directory exists. Press Enter to continue...")
    else:
    	os.mkdir(options.outputDir)

    if options.split:
     input("INFO: you are splitting the dataset. Use this option when validating on same dataset as used for training (ex: QCD or data). Press enter to continue...")
    else:
     input("INFO: you are NOT splitting the dataset. If you are validating on same dataset used for training (ex: QCD or data) add option -s. Otherwise press enter to continue...") 
     
    yamlConfig = parse_config(options.config)	 
    
    print("Getting features...")
    if not options.split:
     X_test, y_test, reco_variables_test, l1_ptcorr_test, weights_test, n_matched_jets = get_features(options, yamlConfig)
     #X_test, y_test, reco_variables_test, l1_ptcorr_test, weights_test = get_features(options, yamlConfig)
    else:
     X_train_val, X_test, y_train_val, y_test, reco_variables_train, reco_variables_test, l1_ptcorr_train, l1_ptcorr_test, weights_train_val, weights_test, n_matched_jets = get_features(options, yamlConfig)

    print("****************************************")
    print("X test shape:",X_test.shape)
    print("Y test shape:",y_test.shape)
    print("Reco jets vars:",reco_variables_test.shape)
    print("L1 jets vars:",l1_ptcorr_test.shape)
    print("Weights shape:",weights_test.shape)
    print("N matched jets:",n_matched_jets.shape)
    print("****************************************")
    
    model = load_model(options.model)
    model.summary()

    l1_pt_predict = getPredictions(X_test,yamlConfig,model) #this is the L1 pT with DNN corrections
    l1_ptcorr =  l1_ptcorr_test[:,0] #this is the L1 pT with standard corrections    
    l1_eta =  l1_ptcorr_test[:,1]
    reco_pt_test = reco_variables_test[:,0] # this is the reco/calo jet pT  
    reco_eta_test = reco_variables_test[:,1] #thie is the reco/calo jet eta  

    plt.figure()
    plt.semilogy()
    plt.hist(l1_ptcorr,200,facecolor='green',alpha=0.75)   
    plt.xlabel('Standard L1 jet pT ')
    plt.savefig(options.outputDir+'/l1jetpt_standard_corrs.png')
        
    cut,cut_regression=0,0
    if options.threshold != 0: cut,cut_regression = options.threshold,options.threshold 
    else: cut,cut_regression = makeRates(options,yamlConfig,model)
        
    if cut != 0 and cut_regression != 0: makeTriggerTurnOn(l1_ptcorr,l1_pt_predict,reco_pt_test,[cut,cut_regression],options)
    makeResolution(reco_pt_test,reco_eta_test,l1_ptcorr,l1_pt_predict,options,"pt")
    makeResolution(reco_pt_test,reco_eta_test,l1_ptcorr,l1_pt_predict,options,"eta")   

    if options.rateHTFile != '':
     
    	if options.HTthreshold != 0: HTcut,HTcut_regression = options.HTthreshold,options.HTthreshold
    	else: HTcut,HTcut_regression = makeHTRates(options,yamlConfig,model)

    	makeHTResolution(n_matched_jets,reco_pt_test,reco_eta_test,l1_ptcorr,l1_eta,l1_pt_predict,options)
    	makeHTTriggerTurnOn(n_matched_jets,l1_ptcorr,l1_pt_predict,reco_pt_test,l1_eta,reco_eta_test,[HTcut,HTcut_regression],options)
           

