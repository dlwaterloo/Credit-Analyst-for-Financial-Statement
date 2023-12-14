import shutil
import os
import json
import pandas as pd
from fastapi import UploadFile
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

def get_document_analysis_client(key, endpoint):
    return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def extract_tables_from_pdf(client, pdf_path):
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", f)
        return poller.result()


def find_table_titles(table, pages):
    candidate_lines = []
    table_top_left = table.bounding_regions[0].polygon[0]

    for page in pages:
        if page.page_number == table.bounding_regions[0].page_number:
            for line in page.lines:
                line_top_y = min(point.y for point in line.polygon)
                distance = table_top_left.y - line_top_y
                if 0 <= distance < 100:
                    candidate_lines.append((line.content, distance))

    candidate_lines.sort(key=lambda x: x[1])
    title_segments = []
    current_segment = []
    previous_y = None
    for line, y in candidate_lines:
        if previous_y is not None and y - previous_y > 15:
            title_segments.append(" ".join(current_segment))
            current_segment = [line]
        else:
            current_segment.append(line)
        previous_y = y
    if current_segment:
        title_segments.append(" ".join(current_segment))

    final_title = " - ".join(title_segments)
    return final_title


def add_table_data(table_data_key, table, tables_data):
    if table_data_key not in tables_data:
        tables_data[table_data_key] = {
            "rows": 0,
            "columns": table.column_count,
            "cells": []
        }
    tables_data[table_data_key]["rows"] += table.row_count

    for cell in table.cells:
        cell_data = {
            "row_index": cell.row_index + tables_data[table_data_key]["rows"] - table.row_count,
            "column_index": cell.column_index,
            "content": cell.content
        }
        tables_data[table_data_key]["cells"].append(cell_data)


def shorten_title_with_gpt(long_title):
    llm = OpenAI(temperature=0)
    prompt_template_str = f"Summarize the following financial statement's table title if exists into a concise, informative title:\n\n{long_title}\n\nShortened Title, else return 'None'"
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run({"long_title": long_title})
    shortened_title = response.strip()
    return shortened_title


def is_balance_sheet(title):
    balance_sheet_phrases = ["balance sheet", "statement of financial position", "report of financial position", "statement of assets and liabilities"]
    return any(phrase in title.lower() for phrase in balance_sheet_phrases)


def is_income_statement(title):
    income_statement_phrases = ["income statement", "profit and loss", "statement of operations"]
    return any(phrase in title.lower() for phrase in income_statement_phrases)


def have_similar_structure(table, existing_table_data):
    # Compare the column count of the new table with the column count stored in the existing table data
    return table.column_count == existing_table_data["columns"]


def process_tables(result):
    tables_data = {}
    balance_sheet_info = None
    income_statement_info = None

    for table_idx, table in enumerate(result.tables):
        current_page_number = table.bounding_regions[0].page_number
        table_title = find_table_titles(table, result.pages)

        if len(table_title) > 350:
            table_title = shorten_title_with_gpt(table_title)

        table_data_key = f"Table {table_idx} - {table_title}" if table_title else f"Table {table_idx}"

        if is_balance_sheet(table_title):
            if balance_sheet_info and have_similar_structure(table, tables_data[balance_sheet_info["key"]]):
                add_table_data(balance_sheet_info["key"], table, tables_data)
            else:
                tables_data[table_data_key] = {
                    "page_number": current_page_number,
                    "rows": 0,
                    "columns": table.column_count,
                    "cells": []
                }
                add_table_data(table_data_key, table, tables_data)
                balance_sheet_info = {"key": table_data_key, "columns": table.column_count}
        elif is_income_statement(table_title):
            if income_statement_info and have_similar_structure(table, tables_data[income_statement_info["key"]]):
                add_table_data(income_statement_info["key"], table, tables_data)
            else:
                tables_data[table_data_key] = {
                    "page_number": current_page_number,
                    "rows": 0,
                    "columns": table.column_count,
                    "cells": []
                }
                add_table_data(table_data_key, table, tables_data)
                income_statement_info = {"key": table_data_key, "columns": table.column_count}
        else:
            tables_data[table_data_key] = {
                "page_number": current_page_number,
                "rows": 0,
                "columns": table.column_count,
                "cells": []
            }
            add_table_data(table_data_key, table, tables_data)

    for table_key in tables_data:
        tables_data[table_key].pop("page_number", None)

    return tables_data


def tables_to_json(tables_data):
    return json.dumps(tables_data, indent=4)


### Start to process the json data

def convert_monetary_value(value):
    value = value.replace('$', '').replace(',', '').strip()
    non_numeric_cases = ['—', '-', '', 'N/A', '—\n:unselected:']
    if value in non_numeric_cases:
        return 0
    if value.startswith('(') and value.endswith(')'):
        value = '-' + value[1:-1]
    elif value.startswith('(') and not value.endswith(')'):
        value = '-' + value[1:]
    elif value.endswith(')') and not value.startswith('('):
        value = '-' + value[1:]
    try:
        return float(value)
    except ValueError:
        return None


def find_best_match(choices, search_term):
    llm = OpenAI(temperature=0.5)
    prompt_template_str = "Find the best match for the label '{search_term}' from this list: {choices}. Please check the list one by one."
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    formatted_choices = ", ".join(choices)
    response = llm_chain.run({"search_term": search_term, "choices": formatted_choices})

    match = response.split(".")[0].strip()

    if match in choices:
        return match

    for choice in choices:
        if search_term.lower() in choice.lower():
            return choice

    return None


def check_label_and_value(table, label_names, key):
    for label_name in label_names:
        label = find_best_match(table.iloc[:, 0].tolist(), label_name)
        if label:
            row_indices = table.index[table.iloc[:, 0] == label].tolist()
            if row_indices:
                for row_index in row_indices:
                    # Iterate through columns from right to left
                    for col_index in range(table.shape[1] - 1, 0, -1):
                        value_str = table.iloc[row_index, col_index]
                        value = convert_monetary_value(value_str)
                        if value != 0:
                            return value
            else:
                print(f"Label '{label}' found, but no corresponding value")
    print(f"Missing '{label_names}' labels")
    return None


def calculate_ratio(numerator, denominator):
    if numerator is not None and denominator is not None and denominator != 0:
        return numerator / denominator
    return None


def extract_company_name_with_langchain(key):
    llm = OpenAI(temperature=0.3)
    prompt_template_str = "Extract the company name or the website name if exists from this table key: '{key}', else return None"
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    response = llm_chain.run({"key": key})
    company_name = response.strip()
    return company_name


def extract_monetary_unit_with_langchain(key):
    llm = OpenAI(temperature=0)
    prompt_template_str = "Extract the monetary unit if exists from this financial table key: '{key}'"
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    response = llm_chain.run({"key": key})
    monetary_unit = response.strip().lower()
    return monetary_unit if monetary_unit in ['thousands', 'millions', 'billions'] else None


def process_balance_sheet(key, data):
    output = {}
    company_name = extract_company_name_with_langchain(key)
    output["Company name"] = company_name if company_name else ""
    
    monetary_unit = extract_monetary_unit_with_langchain(key)
    output["Monetary unit"] = monetary_unit if monetary_unit else ""

    cells = data[key]["cells"]
    df = pd.DataFrame(cells)
    table = df.pivot(index='row_index', columns='column_index', values='content').fillna('')

    try:
        total_current_assets = check_label_and_value(table, ['Total current assets'], key)
        output["Total current assets"] = total_current_assets if total_current_assets is not None else ""
        
        total_current_liabilities = check_label_and_value(table, ['Total current liabilities'], key)
        output["Total current liabilities"] = total_current_liabilities if total_current_liabilities is not None else ""
        
        # Calculate ratios and add to output
        current_ratio = calculate_ratio(total_current_assets, total_current_liabilities)
        output["Current Ratio"] = current_ratio if current_ratio is not None else ""
        
        total_liabilities = check_label_and_value(table, ['Total liabilities'], key)
        output["Total liabilities"] = total_liabilities if total_liabilities is not None else ""
        
        shareholders_equity = check_label_and_value(table, ['Total stockholders\' equity', 'Total equity', 'Total stockholders’ equity (deficit)', 'Total stockholders’ deficit'], key)
        output["Total stockholders' equity"] = shareholders_equity if shareholders_equity is not None else ""
        
        debt_to_equity_ratio = calculate_ratio(total_liabilities, shareholders_equity)
        output["Debt-to-Equity Ratio"] = debt_to_equity_ratio if debt_to_equity_ratio is not None else ""
        
        cash_and_cash_equivalents = check_label_and_value(table, ['Cash and cash equivalents'], key)
        output["Cash and cash equivalents"] = cash_and_cash_equivalents if cash_and_cash_equivalents is not None else ""
        
        cash_ratio = calculate_ratio(cash_and_cash_equivalents, total_current_liabilities)
        output["Cash Ratio"] = cash_ratio if cash_ratio is not None else ""
        
        total_assets = check_label_and_value(table, ['Total assets'], key)
        output["Total Assets"] = total_assets if total_assets is not None else ""

    except Exception as e:
        output["Error"] = str(e)

    return output


def process_income_statement(key, data):
    output = {}
    company_name = extract_company_name_with_langchain(key)
    output["Company name"] = company_name if company_name else ""
    
    monetary_unit = extract_monetary_unit_with_langchain(key)
    output["Monetary unit"] = monetary_unit if monetary_unit else ""

    cells = data[key]["cells"]
    df = pd.DataFrame(cells)
    table = df.pivot(index='row_index', columns='column_index', values='content').fillna('')

    try:
        revenue_labels = ['Revenue', 'Total Revenue', 'Total revenue', "Loss"]
        Revenue = check_label_and_value(table, revenue_labels, key)
        output["Revenue(or Loss)"] = Revenue if Revenue is not None else ""
        
        cost_labels = ['Total costs and expenses', 'Total operating expenses']
        Total_costs = check_label_and_value(table, cost_labels, key)
        output["Total costs"] = Total_costs if Total_costs is not None else ""
        
        EBIT = Revenue - Total_costs if Revenue is not None and Total_costs is not None else None
        output["EBIT"] = EBIT if EBIT is not None else ""
        
        interest_expense = check_label_and_value(table, ['Interest expense', 'Interest income', 'interest expense'], key)
        output["Interest expense (if found)"] = interest_expense if interest_expense is not None else ""
        
        interest_coverage_ratio = calculate_ratio(EBIT, abs(interest_expense)) if interest_expense is not None and interest_expense != 0 else None
        output["Interest Coverage Ratio"] = interest_coverage_ratio if interest_coverage_ratio is not None else ""
        
        net_income_labels = ['Net income', 'Net loss']
        net_money = check_label_and_value(table, net_income_labels, key)
        output["net money (income or loss)"] = net_money if net_money is not None else ""

    except Exception as e:
        output["Error"] = str(e)

    return output


def main(json_data):
    data = json.loads(json_data)
    final_output = {}

    balance_sheet_data = {}
    income_statement_data = {}

    for key in data.keys():
        if ("balance sheet" in key.lower() or
            "financial position" in key.lower() or
            "assets and liabilities" in key.lower()):
            balance_sheet_data[key] = data[key]
        elif ("income statement" in key.lower() or
              "statements of operations" in key.lower() or
              "statement of operations" in key.lower() or
              "statements of income" in key.lower() or
              "profit and loss" in key.lower() or
              "profit or loss" in key.lower() or
              "operating statement" in key.lower() or
              "earnings statement" in key.lower()):
              income_statement_data[key] = data[key]

    if not income_statement_data:
        for key in data.keys():
            if ("statements of comprehensive income" in key.lower() or
                "comprehensive income" in key.lower() or
                "comprehensive net expenditure" in key.lower()):
                income_statement_data[key] = data[key]

    for key in balance_sheet_data:
        balance_output = process_balance_sheet(key, balance_sheet_data)
        final_output.update(balance_output)

    for key in income_statement_data:
        income_output = process_income_statement(key, income_statement_data)
        final_output.update(income_output)

    return final_output


def process_pdf(file: UploadFile):
    # Save the uploaded file to a temporary file
    temp_pdf_path = "temp_pdf.pdf"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    azure_key = os.getenv("AZURE_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")

    os.environ["OPENAI_API_KEY"] = openai_key

    client = get_document_analysis_client(azure_key, azure_endpoint)
    result = extract_tables_from_pdf(client, temp_pdf_path)
    tables_data = process_tables(result)
    json_data = tables_to_json(tables_data)

    # Call the main function with the json_data
    main_output = main(json_data)

    # Optionally delete the temporary file
    os.remove(temp_pdf_path)

    return main_output