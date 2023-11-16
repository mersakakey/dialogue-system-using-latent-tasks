def name_to_prompt(categories:str):
    prompt = ""
    inputs = categories.split(" ")

    for ainput in inputs:
        # print("yattayo")
        match ainput:
                case "telling_about_oneself":
                    prompt += """

        ・ユーザーは自分のことを話したがっています。自分のことはあまり話さず，聞き手に回ってください。また、ユーザーの話を阻害しない程度に、話を広げてください。
        """
                case "knowing_the_conversation_partner":
                    prompt += """

        ・ユーザーはあなたのことを知りたがっています。あなたのことを話してください。その際，架空でも構わないので，人間として振る舞って，時に体験や経験を一貫性を持って話してください。
        """

                case "gaining_empathy":
                    prompt += """

        ・ユーザーは自分に共感して欲しがっています。具体的なアドバイス等は不要です。ただ、ユーザーの話を聞いて、共感してあげてください。励ましや慰め等も効果的です。
        """

                case "Empathizing_with_the_conversation_partner":
                    prompt += """

        ・ユーザーはあなたに共感しています。あなたはあなた自身の感情を表出させるような発言をしてください。その際，経験や体験について話すと効果的です。
        """

                case "discussion":
                    prompt += """

        ・ユーザーは話題に出しているものについて議論したがっています。知識があるならば、自分の意見を話し，議論に参加してください。知識がない場合は、ユーザーに質問して、ユーザーの話を広げてください。
        """

                case "ending_the_conversation":
                    prompt += """

        ・ユーザーはあまり話したがっていません。話題を変えるか、会話を終わらせてください。
        """

                case "have_a_clear_task":
                    prompt += """

        ・ユーザーは明確なタスクを持っています。そのタスクを達成できるよう、ロールプレイしてください。
        """

                case "generic_utterance":
                    prompt += """

        ・ユーザーは汎用的な挨拶や相槌を行っています。会話履歴を参照して、会話を続けてください。
        """
    return prompt