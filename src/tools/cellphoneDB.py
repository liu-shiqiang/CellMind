import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field

from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

from langchain_core.tools import tool

from config.setting import settings

class  CellphoneDBArgs(BaseModel):
    file_path: str = Field(description="Path to the input .h5ad file for cellphoneDB analysis.")

@tool(
    "run_cellphonedb_analysis",
    args_schema=CellphoneDBArgs,

)
def run_cellphonedb_analysis(
    file_path: str,
    work_dir: str,
): 
    # 这是工作路径 ，存放中间文件和结果文件，接下来需要实现中间文件和结果文件的保存
    work = Path(work_dir).expanduser().resolve()
    adata = sc.read_h5ad(file_path, backed='r')

    counts_df = pd.DataFrame(adata.X.toarray(), 
                            index=adata.obs_names, 
                            columns=adata.var_names)

    # get main information
    meta_df = adata.obs[['celltype_l4']].copy() 
    meta_df['Cell'] = adata.obs_names 
    meta_df = meta_df[['Cell', 'celltype_l4']] 

    counts_df.T.to_csv('counts.txt', sep='\t') 
    meta_df.to_csv('meta.txt', sep='\t', index=False)

    cpdb_file_path = '/home/share/huadjyin/home/hebin1/database/cellphonedb/cellphonedb.zip'

    # 运行 CellPhoneDB 分析
    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path=settings.cellphonedb,
        meta_file_path='meta.txt',        
        counts_file_path='counts.txt',    
        counts_data='hgnc_symbol',         
        output_path=settings.OUTPUT_DIR,             
        threshold=0.1,                     
        threads=8,                        
        pvalue=0.05,
        debug_seed=-1,
        score_interactions=True,          
    )

    print("分析完成！结果已保存至 'output' 目录。")
    # 生成可视化图片代码，需要实现


    # CellPhoneDB分析后产生的文件中包含的了分析的结果，下一步还应该实现细胞通讯分析的结果解读
    # 结果实现点：1、meta.txt、counts.txt、等看起来是文本类的结果，用LLM进行解读，查看pubmed_rag代码，加入rag进行解读
    #2、复杂的结果文件，是否可以生成更多可视化图片
    

    # 这个result 返回什么结果，取决于你想让agent看到什么结果
    result = {

    }
    return result
