with open('data/src-dev.txt') as src, open ('data/tgt-dev.txt') as tgt:
    sentences = src.readlines()
    questions = tgt.readlines()
    data = {}
    for idx, (sentence, question) in enumerate(zip(sentences, questions)):
        data[sentence] = data[sentence] + [question] if sentence in data else [question]

    # QF = open('data/src-dev.txt.split', 'w')
    # SF = [open('data/tgt-dev.txt.split{}'.format(x), 'w') for x in range(9)]

    # Repeated sentences

    # for sentence in sentences:
    #     QF.write(sentence)
    #     questions = data[sentence]
    #     for x in range(9):
    #         question = questions[x] if x < len(questions) else '\n'
    #         SF[x].write(question)

    # Unique sentences

    QF = open('data/src-dev.unq.txt', 'w')
    SF = [open('data/tgt-dev.unq.txt.split{}'.format(x), 'w') for x in range(9)]

    for sentence, questions in data.items():
        QF.write(sentence)
        for x in range(9):
            question = questions[x] if x < len(questions) else '\n'
            SF[x].write(question)

    QF.close()
    for x in range(9):
        SF[x].close()
