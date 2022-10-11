from email import header
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, ColumnsAutoSizeMode


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True,
    )

    options.configure_side_bar()
    options.configure_selection("single")
    options.configure_column(field="session", sort="asc")
    options.configure_column(field="water_restriction_number", hide=True, rowGroup=True)
    options.configure_column(field="session_date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    options.configure_column(field="ephys_insertions", dateType="DateType")
    
    # options.configure_column(field="water_restriction_number", header_name="subject", 
    #                          children=[dict(field="water_restriction_number", rowGroup=True),
    #                                    dict(field="session")])
    
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        height=1000,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
    )

    return selection