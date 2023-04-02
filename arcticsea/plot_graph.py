import igraph
import numpy as np



def visualize_temporal(adja_mat,feature_names,full_feature_names,lag_range,all_variable_names,plotname="graph_temporal.pdf"):
    
    if isinstance(adja_mat,np.ndarray):
        adja_mat=adja_mat.tolist()
        
    is_empty_graph=np.sum(np.abs(adja_mat))==0

    gr=igraph.Graph.Weighted_Adjacency(adja_mat, mode="directed", attr="weight", loops=False)
    coord_list=[0]*len(all_variable_names)


    for ell in range(len(feature_names)):
        coord_list[all_variable_names.index(full_feature_names[ell])]=(0.5*ell,0+0.06*(-1)**ell)
        gr.vs[all_variable_names.index(full_feature_names[ell])]["name"]=feature_names[ell]
        for mmm in range(len(lag_range)):
            coord_list[all_variable_names.index(full_feature_names[ell]+'-'+str(lag_range[mmm]))]=(0.5*ell+0.01*(-1)**mmm,0.5*(mmm+1)+0.06*(-1)**ell)
            gr.vs[all_variable_names.index(full_feature_names[ell]+'-'+str(lag_range[mmm]))]["name"]=feature_names[ell]+'_'+str(lag_range[mmm])


    layout = igraph.Layout(coord_list)

    
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"]=6
    visual_style["vertex_color"] = [220,220,220]
    visual_style["vertex_shape"] = "rectangle"
    visual_style["vertex_label"] = gr.vs["name"]
    if not is_empty_graph:
        visual_style["edge_label"] =gr.es["weight"] 
    visual_style["edge_width"] = 1.2
    visual_style["edge_arrow_size"]=0.8
    visual_style["edge_arrow_width"]=0.8
    visual_style["edge_label_size"]=8
    visual_style["layout"] = layout
    visual_style["bbox"] = (800, 800)
    visual_style["margin"] = 40
    igraph.plot(gr,plotname,**visual_style)





def visualize_reduced_graph(adja_mat,feature_names,full_feature_names,lag_range,all_variable_names,plotname="graph_reduced.pdf"):

    if not isinstance(adja_mat,np.ndarray):
        adja_mat=np.array(adja_mat)
    
    d=len(feature_names)
    adja_mat_reduced=np.zeros((d,d))


    n=len(all_variable_names)
    for ell in range(n):
        for mmm in range(n):
            if adja_mat[ell,mmm]!=0:
                
                temp1=all_variable_names[ell]
                temp2=all_variable_names[mmm]
                
                if not temp1 in full_feature_names:
                    if temp1[-3]=='-':
                        temp1=temp1[:-3]
                    else:
                        temp1=temp1[:-2]

                if not temp2 in full_feature_names:
                    if temp2[-3]=='-':
                        temp2=temp2[:-3]
                    else:
                        temp2=temp2[:-2]
                
                adja_mat_reduced[full_feature_names.index(temp1),full_feature_names.index(temp2)]=1
    
    
    gr=igraph.Graph.Weighted_Adjacency(adja_mat_reduced.tolist(), mode="directed", loops=True)
    gr.vs["name"]=feature_names
    layout = gr.layout_fruchterman_reingold()

    visual_style = {}
    visual_style["vertex_size"] = 40
    visual_style["vertex_label_size"]=8
    visual_style["vertex_color"] = [220,220,220]
    visual_style["vertex_shape"] = "rectangle"
    visual_style["vertex_label"] = gr.vs["name"]
    visual_style["edge_width"] = 2
    visual_style["layout"] = layout
    visual_style["bbox"] = (500, 500)
    visual_style["margin"] = 30
    igraph.plot(gr,plotname,**visual_style)
        




def visualize_static(adja_mat,feature_names,plotname="graph_static.pdf"):

    if isinstance(adja_mat,np.ndarray):
        adja_mat=adja_mat.tolist()
        
    is_empty_graph=np.sum(np.abs(adja_mat))==0
        
    gr=igraph.Graph.Weighted_Adjacency(adja_mat, mode="directed", attr="weight", loops=False)
    gr.vs["name"]=feature_names
    layout = gr.layout_kamada_kawai()
    
    visual_style = {}
    visual_style["vertex_size"] = 40
    visual_style["vertex_label_size"]=11
    visual_style["vertex_color"] = [220,220,220]
    visual_style["vertex_shape"] = "rectangle"
    visual_style["vertex_label"] = gr.vs["name"]
    if not is_empty_graph:
        visual_style["edge_label"] =gr.es["weight"]
        visual_style["edge_label_size"]=15
    visual_style["edge_width"] = 2
    visual_style["layout"] = layout
    visual_style["bbox"] = (500, 500)
    visual_style["margin"] = 30
    igraph.plot(gr,plotname,**visual_style)






def visualize_temporal_unweighted(adja_mat,feature_names,full_feature_names,lag_range,all_variable_names,plotname="graph_temporal.pdf"):

    if isinstance(adja_mat,np.ndarray):
        adja_mat=adja_mat.tolist()

    gr=igraph.Graph.Weighted_Adjacency(adja_mat, mode="directed", attr="weight", loops=False)
    coord_list=[0]*len(all_variable_names)

    for ell in range(len(feature_names)):
        coord_list[all_variable_names.index(full_feature_names[ell])]=(0.5*ell,0+0.06*(-1)**ell)
        gr.vs[all_variable_names.index(full_feature_names[ell])]["name"]=feature_names[ell]
        for mmm in range(len(lag_range)):
            coord_list[all_variable_names.index(full_feature_names[ell]+'-'+str(lag_range[mmm]))]=(0.5*ell+0.01*(-1)**mmm,0.5*(mmm+1)+0.06*(-1)**ell)
            gr.vs[all_variable_names.index(full_feature_names[ell]+'-'+str(lag_range[mmm]))]["name"]=feature_names[ell]+'_'+str(lag_range[mmm])

    layout = igraph.Layout(coord_list)

    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"]=6
    visual_style["vertex_color"] = [220,220,220]
    visual_style["vertex_shape"] = "rectangle"
    visual_style["vertex_label"] = gr.vs["name"]
    visual_style["edge_width"] = 1.2
    visual_style["edge_arrow_size"]=0.8
    visual_style["edge_arrow_width"]=0.8
    visual_style["edge_label_size"]=8
    visual_style["layout"] = layout
    visual_style["bbox"] = (800, 800)
    visual_style["margin"] = 40
    igraph.plot(gr,plotname,**visual_style)





def visualize_static_unweighted(adja_mat,feature_names,plotname="graph_static.pdf"):

    if isinstance(adja_mat,np.ndarray):
        adja_mat=adja_mat.tolist()

    is_empty_graph=np.sum(np.abs(adja_mat))==0

    gr=igraph.Graph.Weighted_Adjacency(adja_mat, mode="directed", attr="weight", loops=False)
    gr.vs["name"]=feature_names
    layout = gr.layout_kamada_kawai()

    visual_style = {}
    visual_style["vertex_size"] = 40
    visual_style["vertex_label_size"]=8
    visual_style["vertex_color"] = [220,220,220]
    visual_style["vertex_shape"] = "rectangle"
    visual_style["vertex_label"] = gr.vs["name"]
    visual_style["edge_width"] = 2
    visual_style["layout"] = layout
    visual_style["bbox"] = (500, 500)
    visual_style["margin"] = 30
    igraph.plot(gr,plotname,**visual_style)






# for testing
if __name__ == "__main__":

    aa=np.zeros((12,12))
    aa[0,3]=1
    aa[7,5]=23
    aa[9,2]=-12
    aa[7,10]=9
    aa[10,11]=1

    feature_names=[str(ell) for ell in range(12)]

    visualize_static(aa,feature_names,"static.png")
    visualize_static_unweighted(aa,feature_names,"static_unweighted.png")

    
    lag_range=np.arange(1,4)
    feature_names=['AA','BB','CC']
    full_feature_names=['A','B','C']
    all_variable_names=['A-1','A-2','A-3','A','B','B-1','B-2','B-3','C','C-1','C-2','C-3']
    visualize_temporal(aa,feature_names,full_feature_names,lag_range,all_variable_names,"temporal.png")
    visualize_reduced_graph(aa,feature_names,full_feature_names,lag_range,all_variable_names,"temporal_reduced.png")
    visualize_temporal_unweighted(aa,feature_names,full_feature_names,lag_range,all_variable_names,"temporal_unweighted.png")
