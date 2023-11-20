import React, { useState } from 'react'
import './Page.css';

const Page = () => {

    const [formData, setFormData] = useState({
        no_of_dependents: 0,
        education: 0,
        self_employed: 0,
        income_annum: 0,
        loan_amount: 0,
        loan_term: 0,
        cibil_score: 0,
        residential_assets_value: 0,
        commercial_assets_value: 0,
        luxury_assets_value: 0,
        bank_asset_value: 0,
    });

    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prevData) => ({ ...prevData, [name]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            setPrediction(result.prediction);
        } catch (error) {
            console.error('Error:', error);
        }
    };
    return (
        <>
            <div className="full_page">
                <nav><h2>Loan Approval Prediction</h2></nav>
                <div className="middle_box">
                    <marquee behavior="" direction="">Made by Harsh and Shreeya</marquee>
                    <br />
                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">No of dependents</label>
                            <input type="number" name="no_of_dependents" onChange={handleChange} />
                        </span>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        <span className="input_field">
                            <label htmlFor="">Education</label>
                            <select name="education" id="" onChange={handleChange}>
                                <option value={1}>Graduate</option>
                                <option value={0}>Not Graduate</option>
                            </select>
                        </span>
                    </div>

                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">Employement</label>
                            <select name="self_employed" id="" onChange={handleChange}>
                                <option value={1}>Yes</option>
                                <option value={0}>No</option>
                            </select>
                        </span>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                        <span className="input_field">
                            <label htmlFor="">Income per annum</label>
                            <input type="number" name="income_annum" onChange={handleChange}/>
                        </span>
                    </div>

                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">Loan needed</label>
                            <input type="number" name="loan_amount" onChange={handleChange}/>
                        </span>

                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                        <span className="input_field">
                            <label htmlFor="">Loan Term (in yrs)</label>
                            <input type="number" name="loan_term" onChange={handleChange}/>
                        </span>
                    </div>

                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">Cibil score</label>
                            <input type="number" name="cibil_score" onChange={handleChange}/>
                        </span>

                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                        <span className="input_field">
                            <label htmlFor="">Residential assets value</label>
                            <input type="number" name="residential_assets_value" onChange={handleChange}/>
                        </span>
                    </div>

                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">Commercial assets value</label>
                            <input type="number" name="commercial_assets_value" onChange={handleChange}/>
                        </span>

                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                        <span className="input_field">
                            <label htmlFor="">Luxury assets value</label>
                            <input type="number" name="luxury_assets_value" onChange={handleChange}/>
                        </span>
                    </div>

                    <div className="row_box">
                        <span className="input_field">
                            <label htmlFor="">Bank assets value</label>
                            <input type="number" name="bank_asset_value" onChange={handleChange}/>
                        </span>
                    </div>
                    <br />
                    <button id="submit_btn" onClick={handleSubmit}><b>Submit</b></button>

                <br />
                <div className='pred'><b>Prediction:&nbsp;</b> <div className={prediction === ' Approved' ? 'approved' : 'predval'}>{prediction}</div></div>
                 
                </div>
            </div>
        </>
    )
}

export default Page
