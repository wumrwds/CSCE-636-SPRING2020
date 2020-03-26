# CSCE-636-Project-Part 1

Since we are required to implement a "smart home" for elderly people and patients, I think we'd better consider the negative exceptional actions and facial expressions that indicates that something bad happens, so that our system can react and solve the problems for the users.

### Actions

| Type               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| Slipping           | If we detect our user slip down, we can try to help him stand up. |
| Lying down         | If we detect our user lying down, we can call his family to check whether he is OK or not. |
| Crawling           | If we detect our user is crawling, we can call his family to check whether he is OK or not. |
| Sitting            | If we detect our user sitting for a long while, maybe we should tell him to stand up and take a break. It's good for elder people and patients' health. |
| Walking Upstairs   | If we detect our user is trying to walking upstairs for a long while, maybe we should help him to go upstairs. |
| Walking downstairs | If we detect our user is trying to walking downstairs for a long while, maybe we should help him to go downstairs. |



### Facial Expressions

| Type    | Description                                                  |
| ------- | ------------------------------------------------------------ |
| Disgust | If we detect our user feels disgusted, we might call a doctor for him. |
| Fear    | If we detect our user is scared, we should check whether there's something bad happened. |
| Sad     | If we detect our user is sad for a couple of days, maybe we should check his mental health. |
| Angry   | If we detect our user is angry for a couple of days, maybe we should check his mental health. |

