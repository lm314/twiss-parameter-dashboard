import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
#import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from scipy.stats.qmc import Halton,Sobol
from scipy.stats import norm
from typing import Callable

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

load_figure_template("solar")

class Twiss:
    def __init__(self, alpha: float, beta: float, emit: float):
        self.alpha = alpha
        self.beta = beta
        self.emit = emit

    def __str__(self):
        return f"alpha={self.alpha}\nbeta={self.beta}\nemit={self.emit}"

    def rms_space(self):
        return np.sqrt(self.emit*self.beta)
    
    def rms_angle_waist(self):
        return np.sqrt(self.emit/self.beta)
    
    def gamma(self):
        return (1+self.alpha**2)/self.beta

def ellipse_twiss(twiss_paramters: Twiss, num_points: int = 1000):
    # rmsX = np.sqrt(twiss_paramters.emit/twiss_paramters.beta);
    # rmsTheta = np.sqrt(twiss_paramters.emit*twiss_paramters.beta);
    rmsX = twiss_paramters.rms_space();
    rmsTheta = twiss_paramters.rms_angle_waist();
    
    m = -twiss_paramters.alpha/twiss_paramters.beta
    b = rmsTheta
    a = rmsX
    t = np.linspace(0,2*np.pi,num_points)
    x = a*np.cos(t)
    y = b*np.sin(t)
    x = x.transpose()
    y = y.transpose()
    y = y + x*m
    
    return x,y

def gen_halton_gaussian_4d(twiss_x,twiss_y,num,seed=1,dim=4,loc=[0,0,0,0]):
    rmsX = twiss_x.rms_space();
    rmsThetaX = twiss_x.rms_angle_waist();
    rmsY = twiss_y.rms_space();
    rmsThetaY = twiss_y.rms_angle_waist();    
    
    a=Halton(d=dim,seed=seed)
    vals = a.random(n=num)
    
    dist = norm.ppf(vals,loc=loc,scale=[rmsX,rmsThetaX,rmsY,rmsThetaY])

    mx = -twiss_x.alpha/twiss_x.beta
    my = -twiss_y.alpha/twiss_y.beta;
    
    dist[:,1] += dist[:,0]*mx
    dist[:,3] += dist[:,2]*my
    
    return dist

def gen_dist_normal_coord_4d(func: Callable,**kwarg):
    twiss_x = Twiss(alpha=0,beta=1,emit=1)
    twiss_y = Twiss(alpha=0,beta=1,emit=1)
    return func(twiss_x,twiss_y,**kwarg)

def transform_norm_dist(a: np.ndarray,twiss_x: Twiss,twiss_y: Twiss):
       
    a_copy = a.copy()
    
    twiss_old  = Twiss(alpha=0,beta=1,emit=1)
    
    x_scale = twiss_x.rms_space()/twiss_old.rms_space();
    x_theta_scale = twiss_x.rms_angle_waist()/twiss_old.rms_angle_waist();
    y_scale = twiss_y.rms_space()/twiss_old.rms_space();
    y_theta_scale = twiss_y.rms_angle_waist()/twiss_old.rms_angle_waist();  
    
    mx = -twiss_x.alpha/twiss_x.beta
    my = -twiss_y.alpha/twiss_y.beta;
    
    a_copy[:,0] *= x_scale
    a_copy[:,1] *= x_theta_scale
    a_copy[:,2] *= y_scale
    a_copy[:,3] *= y_theta_scale
    
    a_copy[:,1] += a_copy[:,0]*mx
    a_copy[:,3] += a_copy[:,2]*my
    
    return a_copy

def KE2gamma(KE):
    """Computes the Lorentz Parameter from the kinetic energy
        
    Parameters
    ----------
    KE : array_like
        Kinetic Energy
    Returns
    -------
    array_like
        Lorentz Parameter
    """
    
    mc2 = 0.5109989461e6
    return (1+KE/mc2)
    
def gamma2beta(gamma):
    """Computes the normalized velocity from the kinetic energy
        
    Parameters
    ----------
    gamma : array_like
        Lorentz Parameter
    Returns
    -------
    array_like
        normalized velocity
    """
    
    return np.sqrt(1-1/gamma**2)



def det_plot_scale(ps_df,cutoff = 0.75):
    """Gives the scalings for a physical particle distribution
    
    Takes the 6 dimensional phase space and returns the associated scaling.
    
    Parameters
    ----------
    ps_df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    cutoff : float
        Indicates the cutoff value for scale, e.g. 0.75 means 
        that a value of 750 will return a scale of 3 instead 
        of 0. Must be greater than 0 and equal to or less 
        than 1.
    
    Returns
    -------
    dict of {str : dict of {str : int} and {str : str}}
        A dictionary with the scaling for each dimension. 
        Can be accessed using the the coordinate name at the first 
        level ['x','xp','y','yp','z','delta'].
    """
    
    scaleSteps = 3
    maxExtents = abs(pd.concat([ps_df.max(axis=0), ps_df.min(axis=0)],axis=1))
    maxExtent = maxExtents.max(axis=1)
    maxExtent[maxExtent==0] = 1
    scale = np.floor(np.log10(maxExtent)*cutoff)
    scale = scaleSteps*np.floor(scale/scaleSteps);
    
    scale_info = {name:scale[idx]
              for idx, name in enumerate(ps_df.columns.values)}  

    return scale_info

def make_phase_space_axis_labels(dim,scale_info):
    """Makes labels for axis formatted based on scaling.
    
    Parameters
    ----------
    dim : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension for which the label is returned.
    scale_info : {0,-3,-6,-9,-12,-15}
        Exponential Factor for scaling axis.
        
    
    Returns
    -------
    String
        Label for the specified axis, formatted 
        based on the scaling.
    """
    space_labels = { 0:'(m)',
                    -3:'(mm)',
                    -6:'(\u03BCm)',
                    -9:'(nm)',
                    -12:'(pm)',
                    -15:'(fm)'}
    transverse_angle_labels = {0:'(rad)',
                               -3:'(mrad)',
                               -6:'(\u03BCrad)',
                               -9:'(nrad)',
                               -12:'(prad)',
                               -15:'(frad)'}

    # prevent scalings greater than 0
    scale_info = np.min([0,scale_info])
    
    if(dim=='x'):
            label=f"x {space_labels[scale_info]}"
    elif(dim=='xp'):
            label=f"\u03B8<sub>x</sub> {transverse_angle_labels[scale_info]}"
    if(dim=='y'):
            label=f"y {space_labels[scale_info]}"
    elif(dim=='yp'):
            label=f"\u03B8<sub>y</sub> {transverse_angle_labels[scale_info]}"
    elif(dim=='z'):
            label=f"z {space_labels[scale_info]}"
    elif(dim=='delta'):
            label=f"U03B4 U00D7 10<sup>{int(-scale_info)}</sup>"
            
    return label

def transform_value(value):
    return 10 ** value

particle_dist = gen_dist_normal_coord_4d(gen_halton_gaussian_4d,num=10000)

controls_style = {
    #"position": "fixed",
    #"top": 0,
    #"left": 0,
    #"bottom": 0,
    #"width": "24rem",
    "padding": "1rem 1rem",
    #"background-color": "#f8f9fa",
    "fontSize": "2.0vh",
    "justifyContent": "center",
}

emittance_slider_x = dcc.Slider(0, 3, 0.1,
    value=1,
    id='emittance-slider-x',
    marks={i: {'label': None} for i in range(0,4)}) 

emittance_slider_y = dcc.Slider(0, 3, 0.1,
    value=1,
    id='emittance-slider-y',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(0,4)})  
 
alpha_slider_x = dcc.Slider(-10, 10, 0.1,
    value=0,
    id='alpha-slider-x',
    marks= {i: {'label': f'{i}'} for i in [-10,-5,0,5,10]}) 

alpha_slider_y = dcc.Slider(-10, 10, 0.1,
    value=0,
    id='alpha-slider-y',
    marks= {i: {'label': f'{i}'} for i in [-10,-5,0,5,10]}) 

beta_slider_x = dcc.Slider(-3, 2, 0.001,
    value=0,
    id='beta-slider-x',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(-3,3)})

beta_slider_y = dcc.Slider(-3, 2, 0.001,
    value=0,
    id='beta-slider-y',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(-3,3)})

kinetic_energy_slider = dcc.Slider(0,3,0.1,
    value=1,
    id='kinetic-energy-slider',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(0,4)})

controls = html.Div([
    html.Hr(),
    dbc.Row(dbc.Col(html.P('Normalized Emittance (nm-rad)'))),
    dbc.Row([dbc.Col(html.P(html.P(id='emittance-output-x')),width=3),
            dbc.Col(emittance_slider_x)]),
    dbc.Row([dbc.Col(html.P(id='emittance-output-y'),width=3),
            dbc.Col(emittance_slider_y)]),
    html.Hr(),
    dbc.Row(dbc.Col(html.P('\u03B1'))),
    dbc.Row([dbc.Col(html.P(html.P(id='alpha-output-x')),width=3),
            dbc.Col(alpha_slider_x)]),
    dbc.Row([dbc.Col(html.P(id='alpha-output-y'),width=3),
            dbc.Col(alpha_slider_y)]),
    html.Hr(),
    dbc.Row(dbc.Col(html.P('\u03B2 (m)'))),
    dbc.Row([dbc.Col(html.P(html.P(id='beta-output-x')),width=3),
            dbc.Col(beta_slider_x)]),
    dbc.Row([dbc.Col(html.P(id='beta-output-y'),width=3),
            dbc.Col(beta_slider_y)]),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(id='kinetic-energy-output'))),
    dbc.Row(dbc.Col(kinetic_energy_slider)), 
    html.Hr(),
],
    style=controls_style)

app = Dash(__name__,
           external_stylesheets=[dbc.themes.SOLAR],
           external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" ])

# app.layout = html.Div([
#     dbc.Row([
#         dbc.Col(html.H1('Exploring Twiss Parameters'), style = {'marginLeft':'7px','marginTop':'1rem','marginBottom':'1rem','textAlign':'center'})
#         ]),
#     dbc.Row([
#         dbc.Col(html.Div(dcc.Graph(id = 'graph'),style = {'marginLeft':'15px', 'marginTop':'15px', 'marginRight':'15px'}),
#                 style={'textAlign':'center','display': 'flex',"justifyContent": "center"})
#         ]),
#     dbc.Row([
#         dbc.Col(controls)
#         ])
# ])

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1('Exploring Twiss Parameters'), style = {'marginLeft':'7px','marginTop':'1rem','marginBottom':'1rem','textAlign':'center'})
        ]),
    dbc.Row([
        dbc.Col(controls,width=3),
        dbc.Col(html.Div(dcc.Graph(id = 'graph'),
                         style = {'marginLeft':'15px', 'marginTop':'15px', 'marginRight':'15px'}),
                style={'textAlign':'center','display': 'flex',"justifyContent": "center"},
                width=9)
        ]),
])


@app.callback(
    Output('graph', 'figure'),
    Input('emittance-slider-x', 'value'),
    Input('emittance-slider-y', 'value'),
    Input('alpha-slider-x', 'value'),
    Input('alpha-slider-y', 'value'),
    Input('beta-slider-x', 'value'),
    Input('beta-slider-y', 'value'),
    Input('kinetic-energy-slider', 'value'))
def update_plot(emitn_x,emitn_y,alpha_x,alpha_y,beta_x,beta_y,kinetic_energy):
    
    temp_dist = particle_dist.copy()
    
    # convert from log scale
    emitn_x = transform_value(emitn_x)
    emitn_y = transform_value(emitn_y)
    beta_x = transform_value(beta_x)
    beta_y = transform_value(beta_y)
    kinetic_energy = transform_value(kinetic_energy)
    
    # get geometric emittance
    gamma0 = KE2gamma(kinetic_energy*1e6)
    beta0 = gamma2beta(gamma0)
    emit_x = emitn_x/(beta0*gamma0);
    emit_y = emitn_y/(beta0*gamma0);
        
    # get twiss ellipse
    twiss_x = Twiss(alpha=alpha_x,beta=beta_x,emit=emit_x*1e-9)
    twiss_y = Twiss(alpha=alpha_y,beta=beta_y,emit=emit_y*1e-9)
    x_ell_x,y_ell_x = ellipse_twiss(twiss_x)
    x_ell_y,y_ell_y = ellipse_twiss(twiss_y)
    scalex_x,scaley_x = list(det_plot_scale(pd.DataFrame({'x':x_ell_x,'xp':y_ell_x}),cutoff = 0.9).values())
    scalex_y,scaley_y = list(det_plot_scale(pd.DataFrame({'y':x_ell_y,'yp':y_ell_y}),cutoff = 0.9).values())
    
    #transform particle distribution to new parameters
    temp_dist = transform_norm_dist(temp_dist,twiss_x,twiss_y)
    
    fig = make_subplots(rows=1, 
                        cols=3,
                        subplot_titles=("X Phase Space", "Y Phase Space", "Real-Space Image"))
   
    font_size_plot = 18
    x_standoff = 0
    y_standoff = 0
    
    # -------------X Phase Space--------------------------------  
    trace_ell_x = go.Scatter(
        x=x_ell_x*10**-scalex_x,
        y=y_ell_x*10**-scaley_x,
        # width=600, 
        # height=600,
        # template="solar"
        marker_color='red',
        hoverinfo='skip',
        )
    
    trace_dist_x = go.Histogram2d(
        x=temp_dist[:,0]*10**-scalex_x,
        y=temp_dist[:,1]*10**-scaley_x,
        colorscale="Viridis",
        nbinsx=51,
        nbinsy=51,
        showscale=False,
        hoverinfo='skip',)
    
    fig.add_traces(data=[trace_dist_x,trace_ell_x],rows=1,cols=1)
    fig.update_xaxes(title_text=make_phase_space_axis_labels('x',scalex_x),
                     row=1,
                     col=1,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = x_standoff)
    fig.update_yaxes(title_text=make_phase_space_axis_labels('xp',scaley_x),
                     row=1,
                     col=1,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = y_standoff)
    
    # ---------------------Y phase space-------------------------
    trace_ell_y = go.Scatter(
        x=x_ell_y*10**-scalex_y,
        y=y_ell_y*10**-scaley_y,
        marker_color='red',
        hoverinfo='skip',
        )
    
    trace_dist_y = go.Histogram2d(
        x=temp_dist[:,2]*10**-scalex_y,
        y=temp_dist[:,3]*10**-scaley_y,
        colorscale="Viridis",
        nbinsx=51,
        nbinsy=51,
        showscale=False,
        hoverinfo='skip',)
    
    fig.add_traces(data=[trace_dist_y,trace_ell_y],rows=1,cols=2)
    fig.update_xaxes(title_text=make_phase_space_axis_labels('y',scalex_y),
                     row=1,
                     col=2,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = x_standoff)
    fig.update_yaxes(title_text=make_phase_space_axis_labels('yp',scaley_y),
                     row=1,
                     col=2,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = y_standoff)
    
    # ------------------real space image-------------------------------
    trace_image = go.Histogram2d(
        x=temp_dist[:,0]*10**-scalex_x,
        y=temp_dist[:,2]*10**-scalex_y,
        colorscale="Viridis",
        nbinsx=51,
        nbinsy=51,
        showscale=False,
        hoverinfo='skip',)
    
    fig.add_traces(data=[trace_image],rows=1,cols=3)
    fig.update_xaxes(title_text=make_phase_space_axis_labels('x',scalex_x),
                     row=1,
                     col=3,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = x_standoff)
    fig.update_yaxes(title_text=make_phase_space_axis_labels('y',scalex_y),
                     row=1,
                     col=3,
                     titlefont=dict(size=font_size_plot),
                     title_standoff = y_standoff)

    #--------------------overall plot---------------------------------------
    
    fig.update_layout(showlegend=False,
                        autosize=True,
                        minreducedwidth=250,
                        minreducedheight=250,
                        width=1200,
                        height=450,
                        font_size=font_size_plot)
    
    fig.update_annotations(font_size=font_size_plot)
    
    return fig

@app.callback(
    Output(component_id='emittance-output-x', component_property='children'),
    Input('emittance-slider-x', 'value' ))
def update_emittance_x(value):
    return f"x: {transform_value(value):.1f}"

@app.callback(
    Output(component_id='emittance-output-y', component_property='children'),
    Input('emittance-slider-y', 'value' ))
def update_emittance_y(value):
    return f"y: {transform_value(value):.1f}"

@app.callback(
    Output(component_id='alpha-output-x', component_property='children'),
    Input('alpha-slider-x', 'value' ))
def update_alpha_x(value):
    return f"x: {value:.1f}"

@app.callback(
    Output(component_id='alpha-output-y', component_property='children'),
    Input('alpha-slider-y', 'value' ))
def update_alpha_y(value):
    return f"y: {value:.1f}"

@app.callback(
    Output(component_id='beta-output-x', component_property='children'),
    Input('beta-slider-x', 'value' ))
def update_beta_x(value):
    return f"x: {transform_value(value):.3f}"

@app.callback(
    Output(component_id='beta-output-y', component_property='children'),
    Input('beta-slider-y', 'value' ))
def update_beta_y(value):
    return f"y: {transform_value(value):.3f}"

@app.callback(
    Output(component_id='kinetic-energy-output', component_property='children'),
    Input('kinetic-energy-slider', 'value' ))
def update_kinetic_energy(value):
    return f"Kinetic Energy (MeV): {transform_value(value):.0f}"

if __name__ == '__main__':
    app.run_server(debug=True)