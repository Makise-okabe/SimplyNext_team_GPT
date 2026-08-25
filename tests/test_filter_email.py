from career_agent.nodes.filter_email import filter_email


def test_talentconnect_sender_is_career_email():
    state = {
        "email": {
            "sender_email": "nustalentconnect@csm.symplicity.com",
            "subject": "New and Trending Jobs",
        }
    }
    result = filter_email(state)
    assert result["is_career_email"] is True
