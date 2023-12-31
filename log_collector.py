import csv

import requests

url = 'https://politiloggen-vis-frontend.bks-prod.politiet.no/api/messagethread'
headers = {
    'Accept': '*/*',
    'Accept-Language': 'nb-NO,nb;q=0.9,no;q=0.8,nn;q=0.7,en-US;q=0.6,en;q=0.5,de;q=0.4',
    'Connection': 'keep-alive',
    'Content-type': 'application/json; charset=UTF-8',
    'DNT': '1',
    'Origin': 'https://www.politiet.no',
    'Referer': 'https://www.politiet.no/politiloggen/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

max_take = 1000
json_data = {
    'sortByEnum': 'Date',
    'sortByAsc': False,
    'timeSpanType': 'Custom',
    'dateTimeFrom': '2023-10-26T00:00:00.000Z',
    'dateTimeTo': '2023-11-08T12:00:00.000Z',
    'skip': 0,
    'take': max_take,
}

fields = ['parent_id', 'district', 'districtId', 'category', 'municipality', 'message_id', 'text', 'createdOn',
          'updatedOn']


def fetch_data():
    list_dict = []
    while True:
        response = requests.post('https://politiloggen-vis-frontend.bks-prod.politiet.no/api/messagethread',
                                 headers=headers, json=json_data, )

        if len(response.json()['messageThreads']) < max_take:
            print("ok")
            len(response.json()['messageThreads'])
            break

        if 'messageThreads' not in response.json():
            print("aaa")
        for line in response.json()['messageThreads']:
            for msg in line['messages']:
                list_dict.append({
                    fields[0]: line['id'],
                    fields[1]: line[fields[1]],
                    fields[2]: line[fields[2]],
                    fields[3]: line[fields[3]],
                    fields[4]: line[fields[4]],
                    fields[5]: msg['id'],
                    fields[6]: msg[fields[6]],
                    fields[7]: msg[fields[7]],
                })
        json_data['skip'] += max_take
    return list_dict


data_dict = fetch_data()

with open(f"logs{json_data.get('dateTimeTo')}.csv", 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data_dict)
