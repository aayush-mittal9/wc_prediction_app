import streamlit as st
import logging

from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from io import StringIO

from vis_utils import DataStore


logging.basicConfig(level=logging.INFO, format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

st.set_page_config(layout="wide")

def main():
    
    ds = DataStore()

    st.title("PyGWc")
    

    with st.sidebar.form(key="control_form"):
        id_ = st.multiselect("ID", options=sorted(ds.df.id.unique()))
        variable_ = st.multiselect("Columns", options=ds.df.columns)
        target_ = st.selectbox("Target", options=ds.df.columns)
        date_ = st.date_input('Date Range', value=(datetime(2021, 3, 2, 12, 10), datetime(2021, 3, 2, 12, 22)))
        cluster_variable_ = st.selectbox("Cluster Variable", options=ds.df.columns)
        cluster_number_ = st.selectbox("Number of Clusters",options=list(range(2,17)))
        is_submit = st.form_submit_button("Enter")

    params = {"id_":id_, "variable_":variable_, "target_":target_, "date_":date_, "cluster_variable_":cluster_variable_, "cluster_number_":cluster_number_}
    ds.build_dataset(**params)

    st.dataframe(ds.df)

    if is_submit:

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Variable","Correlation","Prediction","Forecast","Clustering"])

        # creating a single-element container.
        placeholder = st.empty()

        with placeholder.container():
            
                for fig in ds.plot_variable():
                    tab1.plotly_chart(fig)
                for fig in ds.plot_correlation_matrix():
                    tab2.plotly_chart(fig)
                for fig in ds.plot_facet_grid():
                    tab2.pyplot(fig)
                for fig in ds.plot_catplot():
                    tab2.pyplot(fig)
                tab2.pyplot(ds.plot_kde_plot())
                for fig in ds.plot_corr_dist_plot():
                    tab2.pyplot(fig)
                ds.plot_prediction(tab3)
                ds.visualize_forecast(tab4)
                ds.visualize_forecast_cluster(tab5)
        
            



if __name__ == "__main__":
    main()