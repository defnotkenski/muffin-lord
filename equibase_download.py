# pip install requests
import sys

import requests
import json

url = "https://www.equibase.com/premium/eebCustomerLogon.cfm?TMP=customeradminmain.cfm&QS=logon%3DY"
apikey = "aef5066ff752b64c817d6ee8fb53e52a6859924a"
params = {
    "url": url,
    "apikey": apikey,
    "js_render": "true",
    # "session_id": "22345",
    "premium_proxy": "true",
    "js_instructions": json.dumps(
        [
            {"fill": ["//input[@name='user_id']", "hikenny@me.com"]},
            {"fill": ["//input[@id='customer_password']", "muffinsoda"]},
            {"click": "/html/body/section[5]/div/div[1]/div/div/div/div/form/div/div[3]/div/input"},
        ]
    ),
}


response_login = requests.get("https://api.zenrows.com/v1/", params=params)

# print(response_login.text)
# print(response_login.headers)

# headers_dict = response_login.headers

# equibase_headers = {
#     "Content-Encoding": headers_dict["Zr-Content-Encoding"],
#     "Content-Type": headers_dict["Zr-Content-Type"],
#     "Cookies": headers_dict["Zr-Cookies"],
#     "Final-Url": headers_dict["Zr-Final-Url"],
# }

url_download = "https://www.equibase.com/premium/eqpTMResultChartDownload.cfm?tid=82270004&seq=1"

params_download = {
    "url": url_download,
    "apikey": apikey,
    "js_render": "true",
    # "session_id": "22345",
    "custom_headers": "true",
    # "premium_proxy": "true",
    "js_instructions": json.dumps(
        [
            {"wait_for": "//a[div[contains(text(), 'XML Format')]]"},
            {"click": "//a[div[contains(text(), 'XML Format')]]"},
        ]
    ),
}

response_download = requests.get(
    "https://api.zenrows.com/v1/",
    params=params_download,
    # headers=equibase_headers,
)
print(response_download.text)


# if response_download.status_code == 200:
#     with open("output.xml", "wb") as f:
#         f.write(response_download.content)
#     print("File downloaded and saved successfully!")
# else:
#     print(f"Failed to download the file. Status code: {response_download.text}")

print("")
