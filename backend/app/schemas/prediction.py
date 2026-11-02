
from pydantic import BaseModel, Field, ConfigDict


class IVFPredictionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Female Age": 34,
                "AMH": 2.5,
                "Male Age": 36,
                "Sperm count": 35,
                "Sperm motility": 45,
                "Sperm morphology": 0.05,
                "Oocyte Maturity MII": 8,
                "Oocyte Maturity MI": 2,
                "Oocyte Maturity others": 1,
                "Number of fertilized eggs": 6,
                "Embryo Grade 1": 3,
                "Embryo Grade 2": 2,
                "Embryo Grade 3": 0,
                "Embryo Grade 4": 0,
                "Number of embryos": 2
            }
        }
    )

    female_age: float = Field(alias="Female Age", ge=18, le=60)
    amh: float = Field(alias="AMH", ge=0)
    male_age: float = Field(alias="Male Age", ge=18, le=100)
    sperm_count: float = Field(alias="Sperm count", ge=0)
    sperm_motility: float = Field(alias="Sperm motility", ge=0)
    sperm_morphology: float = Field(alias="Sperm morphology", ge=0)

    oocyte_mii: float = Field(alias="Oocyte Maturity MII", ge=0)
    oocyte_mi: float = Field(alias="Oocyte Maturity MI", ge=0)
    oocyte_others: float = Field(alias="Oocyte Maturity others", ge=0)

    fertilized_eggs: float = Field(alias="Number of fertilized eggs", ge=0)

    embryo_grade_1: float = Field(alias="Embryo Grade 1", ge=0)
    embryo_grade_2: float = Field(alias="Embryo Grade 2", ge=0)
    embryo_grade_3: float = Field(alias="Embryo Grade 3", ge=0)
    embryo_grade_4: float = Field(alias="Embryo Grade 4", ge=0)

    number_of_embryos: float = Field(alias="Number of embryos", ge=0)
